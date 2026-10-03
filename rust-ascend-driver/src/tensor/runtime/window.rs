use super::indexing::{allocate, layout};
use super::{AscendRuntime, ComputeClient, Result, TensorBuffer, WORKER, error};
use crate::tensor::{DType, TensorLayout, TokenWindow, window::WindowPlan};
use ruda_core::tensor::Shape;

fn launch(
    client: &ComputeClient<AscendRuntime>,
    plan: WindowPlan,
    tensors: [&TensorBuffer; 2],
    backward: bool,
) -> Result<()> {
    client.flush().map_err(error)?;
    let layouts = if backward {
        [&plan.output, &plan.source]
    } else {
        [&plan.source, &plan.output]
    };
    let guards = tensors
        .iter()
        .map(|t| client.get_resource(t.handle.clone()).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let resources = guards
        .iter()
        .zip(layouts)
        .map(|(guard, layout)| {
            let mut resource = guard.resource().clone();
            if resource.byte_len() < layout.byte_len() {
                return Err(error("token window resource is too short"));
            }
            resource.size = layout.byte_len();
            Ok(resource)
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| error("token window binding count mismatch"))?;
    let worker = &WORKER
        .get()
        .ok_or_else(|| error("Ascend runtime is not initialized"))?
        .1;
    let result = worker.call(move |state| state.token_window(plan, resources, backward));
    drop(guards);
    result
}

pub(super) fn forward(
    client: &ComputeClient<AscendRuntime>,
    input: TensorBuffer,
    window: TokenWindow,
) -> Result<TensorBuffer> {
    let plan = WindowPlan::new(&layout(&input)?, window)?;
    let output = allocate(client, &plan.output);
    if window.count != 0 {
        launch(client, plan, [&input, &output], false)?;
    }
    Ok(output)
}

pub(super) fn backward(
    client: &ComputeClient<AscendRuntime>,
    shape: Shape,
    grad: TensorBuffer,
    window: TokenWindow,
) -> Result<TensorBuffer> {
    let shape = shape
        .iter()
        .map(|&n| i64::try_from(n).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let plan = WindowPlan::new(&TensorLayout::contiguous(&shape, DType::F32)?, window)?;
    if layout(&grad)? != plan.output {
        return Err(error("token window gradient shape/dtype mismatch"));
    }
    let output = allocate(client, &plan.source);
    if plan.source.byte_len() != 0 {
        launch(client, plan, [&grad, &output], true)?;
    }
    Ok(output)
}
