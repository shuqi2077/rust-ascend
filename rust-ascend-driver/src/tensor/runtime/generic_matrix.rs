use super::{AscendRuntime, ComputeClient, Result, TensorBuffer, WORKER, error, indexing};
use crate::tensor::{DType, TensorLayout, layout::matmul_shape};

pub(super) fn output_layout(a: &TensorLayout, b: &TensorLayout) -> Result<TensorLayout> {
    if a.dtype() != DType::F32
        || b.dtype() != DType::F32
        || !(2..=6).contains(&a.shape().len())
        || !(2..=6).contains(&b.shape().len())
    {
        return Err(error("generic Ascend matmul requires FP32 rank 2..6"));
    }
    TensorLayout::contiguous(&matmul_shape(a.shape(), b.shape())?, DType::F32)
}

pub(super) fn matmul(
    client: &ComputeClient<AscendRuntime>,
    a: TensorBuffer,
    b: TensorBuffer,
) -> Result<TensorBuffer> {
    let a_layout = indexing::layout(&a)?;
    let b_layout = indexing::layout(&b)?;
    let out_layout = output_layout(&a_layout, &b_layout)?;
    let out = indexing::allocate(client, &out_layout);
    client.flush().map_err(error)?;
    let layouts = [a_layout, b_layout, out_layout];
    let guards = [&a, &b, &out]
        .iter()
        .map(|t| client.get_resource(t.handle.clone()).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let resources = guards
        .iter()
        .zip(&layouts)
        .map(|(guard, layout)| {
            let mut resource = guard.resource().clone();
            if resource.byte_len() < layout.byte_len() {
                return Err(error("matmul resource is too short"));
            }
            resource.size = layout.byte_len();
            Ok(resource)
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| error("matmul binding count mismatch"))?;
    let result = WORKER
        .get()
        .ok_or_else(|| error("Ascend runtime is not initialized"))?
        .1
        .call(move |state| state.matmul_fp32(layouts, resources));
    drop(guards);
    result?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp32_broadcast_and_empty_contract() {
        let layout = |shape: &[i64]| TensorLayout::contiguous(shape, DType::F32).unwrap();
        assert_eq!(
            output_layout(&layout(&[2, 1, 3, 5]), &layout(&[1, 4, 5, 7]))
                .unwrap()
                .shape(),
            &[2, 4, 3, 7]
        );
        assert_eq!(
            output_layout(&layout(&[0, 5]), &layout(&[5, 7]))
                .unwrap()
                .byte_len(),
            0
        );
        assert_eq!(
            output_layout(&layout(&[3, 0]), &layout(&[0, 7]))
                .unwrap()
                .shape(),
            &[3, 7]
        );
        assert!(output_layout(&layout(&[3, 5]), &layout(&[4, 7])).is_err());
        assert!(
            output_layout(
                &layout(&[3, 5]),
                &TensorLayout::contiguous(&[5, 7], DType::BF16).unwrap()
            )
            .is_err()
        );
    }
}
