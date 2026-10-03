use super::{Primitive, Result, buffer};
pub use crate::runtime::TokenWindow;
use crate::{
    Ascend, Autodiff,
    driver::CannError,
    runtime::{AscendDevice, AscendRuntime, ComputeClient},
};
use ruda_autodiff::{
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use ruda_core::tensor::{Metadata, Shape};
use ruda_tensor::{
    Backend, TensorPrimitive,
    api::{Int, Tensor},
    tensor::{FloatTensor, IntTensor},
};

pub trait TokenWindowBackend: Backend {
    fn token_window(input: FloatTensor<Self>, window: TokenWindow) -> Result<FloatTensor<Self>>;
    fn token_window_int(input: IntTensor<Self>, window: TokenWindow) -> Result<IntTensor<Self>>;
}

/// Select flattened rows from a contiguous FP32 [B,T,H] time window.
/// Backward scatters to original token positions; shared/chunk gradients accumulate in RUDA.
pub fn token_window<B: TokenWindowBackend>(
    input: Tensor<B, 3>,
    window: TokenWindow,
) -> Result<Tensor<B, 2>> {
    let input = match input.into_primitive() {
        TensorPrimitive::Float(input) => input,
        TensorPrimitive::QFloat(_) => {
            return Err(CannError::InvalidTensor(
                "token window requires unquantized FP32".into(),
            ));
        }
    };
    B::token_window(input, window).map(|out| Tensor::from_primitive(TensorPrimitive::Float(out)))
}

/// Same bit-preserving window for integer token/label tensors [B,T,H].
pub fn token_window_int<B: TokenWindowBackend>(
    input: Tensor<B, 3, Int>,
    window: TokenWindow,
) -> Result<Tensor<B, 2, Int>> {
    B::token_window_int(input.into_primitive(), window).map(Tensor::from_primitive)
}

fn forward(input: Primitive, window: TokenWindow) -> Result<Primitive> {
    let client = input.client.clone();
    let device = input.device.clone();
    let out = AscendRuntime::token_window(&client, buffer(input), window)?;
    Ok(Primitive::new(
        client,
        out.handle,
        Metadata::new(out.shape, out.strides),
        device,
        out.dtype,
    ))
}

fn float_forward(input: Primitive, window: TokenWindow) -> Result<Primitive> {
    if input.dtype != crate::tensor::DType::F32 {
        return Err(CannError::InvalidTensor(
            "differentiable token window requires FP32".into(),
        ));
    }
    forward(input, window)
}

impl TokenWindowBackend for Ascend {
    fn token_window(input: FloatTensor<Self>, window: TokenWindow) -> Result<FloatTensor<Self>> {
        float_forward(input, window)
    }
    fn token_window_int(input: IntTensor<Self>, window: TokenWindow) -> Result<IntTensor<Self>> {
        forward(input, window)
    }
}

#[derive(Clone)]
struct WindowState {
    shape: Shape,
    window: TokenWindow,
    client: ComputeClient<AscendRuntime>,
    device: AscendDevice,
}
impl std::fmt::Debug for WindowState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowState")
            .field("shape", &self.shape)
            .field("window", &self.window)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
struct WindowBackward;
impl<B: crate::backend::AscendTensorBackend> Backward<B, 1> for WindowBackward {
    type State = WindowState;
    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let WindowState {
            shape,
            window,
            client,
            device,
        } = ops.state;
        let grad = grads.consume::<B>(&ops.node);
        assert!(
            grad.device == device && grad.client.same_execution_queue(&client),
            "token window gradient device/queue mismatch"
        );
        let out = AscendRuntime::token_window_backward(&client, shape, buffer(grad), window)
            .expect("Ascend token window backward failed");
        if let Some(parent) = ops.parents[0].as_ref() {
            grads.register::<B>(
                parent.id,
                Primitive::new(
                    client,
                    out.handle,
                    Metadata::new(out.shape, out.strides),
                    device,
                    out.dtype,
                ),
            );
        }
    }
}
impl<B: crate::backend::AscendTensorBackend, C: CheckpointStrategy> TokenWindowBackend
    for Autodiff<B, C>
{
    fn token_window(input: FloatTensor<Self>, window: TokenWindow) -> Result<FloatTensor<Self>> {
        let state = WindowState {
            shape: input.primitive.meta.shape().clone(),
            window,
            client: input.primitive.client.clone(),
            device: input.primitive.device.clone(),
        };
        let output = float_forward(input.primitive, window)?;
        Ok(
            match WindowBackward
                .prepare::<C>(WindowBackward, [input.node])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => prep.finish(state, output),
                OpsKind::UnTracked(prep) => prep.finish(output),
            },
        )
    }
    fn token_window_int(input: IntTensor<Self>, window: TokenWindow) -> Result<IntTensor<Self>> {
        forward(input, window)
    }
}
impl TokenWindowBackend for crate::RudaAscend {
    fn token_window(input: FloatTensor<Self>, window: TokenWindow) -> Result<FloatTensor<Self>> {
        float_forward(input, window)
    }
    fn token_window_int(input: IntTensor<Self>, window: TokenWindow) -> Result<IntTensor<Self>> {
        forward(input, window)
    }
}
