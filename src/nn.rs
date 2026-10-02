//! Native Ascend operations on RUDA tensors and RUDA's existing autodiff graph.
use crate::{Ascend, Autodiff, driver::CannError, runtime::{AscendRuntime, TensorBuffer}};
use ruda_autodiff::{checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients, ops::{Backward, Ops, OpsKind}};
use ruda_core::tensor::Metadata;
use ruda_tensor::{Backend, TensorPrimitive, api::Tensor, tensor::FloatTensor};
use ruda_tensor_device::RudaTensor;

type Primitive = RudaTensor<AscendRuntime>;
type Result<T> = std::result::Result<T, CannError>;

/// Backend extension for native common-IR RMSNorm.
pub trait RmsNormBackend: Backend {
    /// Forward primitive, registering first-order derivatives when this backend tracks them.
    fn rms_norm(input: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f64)
        -> Result<FloatTensor<Self>>;
}

/// Normalize the last dimension of a contiguous FP32 RUDA tensor.
/// The shared weight has rank one; width must be 32..4096 and divisible by 32.
/// `Autodiff<Ascend>` tracks input and weight gradients using saved device statistics.
pub fn rms_norm<B: RmsNormBackend, const D: usize>(input: Tensor<B, D>, weight: Tensor<B, 1>,
    epsilon: f64) -> Result<Tensor<B, D>> {
    let unquantized = |primitive: TensorPrimitive<B>| -> Result<FloatTensor<B>> { match primitive {
        TensorPrimitive::Float(tensor) => Ok(tensor),
        TensorPrimitive::QFloat(_) => Err(CannError::InvalidTensor("native RMSNorm does not dequantize inputs implicitly".into())),
    }};
    let input = unquantized(input.into_primitive())?;
    let weight = unquantized(weight.into_primitive())?;
    B::rms_norm(input, weight, epsilon).map(|output| Tensor::from_primitive(TensorPrimitive::Float(output)))
}

fn buffer(tensor: Primitive) -> TensorBuffer {
    TensorBuffer { handle: tensor.handle, shape: tensor.meta.shape().clone(),
        strides: tensor.meta.strides().clone(), dtype: tensor.dtype }
}
fn check_queue(input: &Primitive, others: &[&Primitive]) -> Result<()> {
    for other in others {
        if input.device != other.device || !input.client.same_execution_queue(&other.client) {
            return Err(CannError::InvalidTensor("native RMSNorm device or execution queue mismatch".into()));
        }
    }
    Ok(())
}
fn forward(input: Primitive, weight: Primitive, epsilon: f64) -> Result<[Primitive; 2]> {
    check_queue(&input, &[&weight])?;
    let client = input.client.clone(); let device = input.device.clone();
    let output = AscendRuntime::rms_norm(&client, buffer(input), buffer(weight), epsilon)?;
    Ok(output.map(|b| Primitive::new(client.clone(), b.handle, Metadata::new(b.shape,b.strides),
        device.clone(), b.dtype)))
}
fn backward(input: Primitive, weight: Primitive, grad: Primitive, rstd: Primitive) -> Result<[Primitive; 2]> {
    check_queue(&input, &[&weight,&grad,&rstd])?;
    let client = input.client.clone(); let device = input.device.clone();
    let output = AscendRuntime::rms_norm_backward(&client, buffer(input), buffer(weight), buffer(grad), buffer(rstd))?;
    Ok(output.map(|b| Primitive::new(client.clone(), b.handle, Metadata::new(b.shape,b.strides),
        device.clone(), b.dtype)))
}

impl RmsNormBackend for Ascend {
    fn rms_norm(input: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f64) -> Result<FloatTensor<Self>> {
        forward(input, weight, epsilon).map(|[output,_]| output)
    }
}

#[derive(Debug)]
struct RmsNormBackward;
impl Backward<Ascend, 2> for RmsNormBackward {
    type State = (Primitive, Primitive, Primitive);
    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
        let (input, weight, rstd) = ops.state;
        let grad = grads.consume::<Ascend>(&ops.node);
        let output = backward(input, weight, grad, rstd).expect("Ascend RMSNorm backward failed");
        for (parent, gradient) in ops.parents.into_iter().zip(output) {
            if let Some(parent) = parent { grads.register::<Ascend>(parent.id, gradient); }
        }
    }
}

impl<C: CheckpointStrategy> RmsNormBackend for Autodiff<Ascend, C> {
    fn rms_norm(input: FloatTensor<Self>, weight: FloatTensor<Self>, epsilon: f64) -> Result<FloatTensor<Self>> {
        let x = input.primitive.clone(); let w = weight.primitive.clone();
        let [output,rstd] = forward(x.clone(), w.clone(), epsilon)?;
        Ok(match RmsNormBackward.prepare::<C>([input.node, weight.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((x,w,rstd), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }
}
