//! Native Ascend operations on RUDA tensors and RUDA's existing autodiff graph.
use crate::{Ascend, Autodiff, driver::CannError, runtime::{AscendRuntime, TensorBuffer}};
use ruda_autodiff::{checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients, ops::{Backward, Ops, OpsKind}};
use ruda_core::tensor::{Metadata, Shape};
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

/// Backend extension for native last-axis Softmax and LogSoftmax.
pub trait SoftmaxBackend: Backend {
    /// Forward primitive with optional log-probabilities and native first-order backward.
    fn normalized_exponential(input: FloatTensor<Self>, logarithmic: bool) -> Result<FloatTensor<Self>>;
}

/// Backend extension for native last-axis sum and mean with a retained size-one axis.
pub trait ReductionBackend: Backend {
    fn reduce_last(input: FloatTensor<Self>, mean: bool) -> Result<FloatTensor<Self>>;
}

/// Native FP32 last-axis sum on a RUDA tensor, including first-order autodiff.
pub fn sum_last<B: ReductionBackend, const D: usize>(input: Tensor<B,D>) -> Result<Tensor<B,D>> {
    reduce_last(input,false)
}

/// Native FP32 last-axis mean on a RUDA tensor, including first-order autodiff.
pub fn mean_last<B: ReductionBackend, const D: usize>(input: Tensor<B,D>) -> Result<Tensor<B,D>> {
    reduce_last(input,true)
}

fn reduce_last<B: ReductionBackend, const D: usize>(input: Tensor<B,D>, mean: bool) -> Result<Tensor<B,D>> {
    let primitive=match input.into_primitive() {
        TensorPrimitive::Float(tensor)=>tensor,
        TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("native reduction does not dequantize inputs implicitly".into())),
    };
    B::reduce_last(primitive,mean).map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Native last-axis FP32 Softmax on a RUDA tensor, including RUDA autodiff.
pub fn softmax<B: SoftmaxBackend, const D: usize>(input: Tensor<B,D>) -> Result<Tensor<B,D>> {
    normalized_exponential(input, false)
}

/// Native last-axis FP32 LogSoftmax on a RUDA tensor, including RUDA autodiff.
pub fn log_softmax<B: SoftmaxBackend, const D: usize>(input: Tensor<B,D>) -> Result<Tensor<B,D>> {
    normalized_exponential(input, true)
}

fn normalized_exponential<B: SoftmaxBackend, const D: usize>(input: Tensor<B,D>, logarithmic: bool)
    -> Result<Tensor<B,D>> {
    let primitive = match input.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor,
        TensorPrimitive::QFloat(_) => return Err(CannError::InvalidTensor("native Softmax does not dequantize inputs implicitly".into())),
    };
    B::normalized_exponential(primitive, logarithmic).map(|output| Tensor::from_primitive(TensorPrimitive::Float(output)))
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
fn check_queue(input: &Primitive, others: &[&Primitive], operation: &str) -> Result<()> {
    for other in others {
        if input.device != other.device || !input.client.same_execution_queue(&other.client) {
            return Err(CannError::InvalidTensor(format!("native {operation} device or execution queue mismatch")));
        }
    }
    Ok(())
}
fn forward(input: Primitive, weight: Primitive, epsilon: f64) -> Result<[Primitive; 2]> {
    check_queue(&input, &[&weight], "RMSNorm")?;
    let client = input.client.clone(); let device = input.device.clone();
    let output = AscendRuntime::rms_norm(&client, buffer(input), buffer(weight), epsilon)?;
    Ok(output.map(|b| Primitive::new(client.clone(), b.handle, Metadata::new(b.shape,b.strides),
        device.clone(), b.dtype)))
}
fn backward(input: Primitive, weight: Primitive, grad: Primitive, rstd: Primitive) -> Result<[Primitive; 2]> {
    check_queue(&input, &[&weight,&grad,&rstd], "RMSNorm")?;
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

fn softmax_forward(input: Primitive, logarithmic: bool) -> Result<Primitive> {
    let client = input.client.clone(); let device = input.device.clone();
    let b = if logarithmic {AscendRuntime::log_softmax(&client, buffer(input))?}
        else {AscendRuntime::softmax(&client, buffer(input))?};
    Ok(Primitive::new(client, b.handle, Metadata::new(b.shape,b.strides), device, b.dtype))
}
fn softmax_backward(output: Primitive, grad: Primitive, logarithmic: bool) -> Result<Primitive> {
    check_queue(&output, &[&grad], if logarithmic {"LogSoftmax"} else {"Softmax"})?;
    let client = output.client.clone(); let device = output.device.clone();
    let b = if logarithmic {AscendRuntime::log_softmax_backward(&client, buffer(output), buffer(grad))?}
        else {AscendRuntime::softmax_backward(&client, buffer(output), buffer(grad))?};
    Ok(Primitive::new(client, b.handle, Metadata::new(b.shape,b.strides), device, b.dtype))
}
impl SoftmaxBackend for Ascend {
    fn normalized_exponential(input: FloatTensor<Self>, logarithmic: bool) -> Result<FloatTensor<Self>> {
        softmax_forward(input, logarithmic)
    }
}

#[derive(Debug)]
struct SoftmaxBackward;
impl Backward<Ascend, 1> for SoftmaxBackward {
    type State = (Primitive, bool);
    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let (output, logarithmic) = ops.state;
        let grad = grads.consume::<Ascend>(&ops.node);
        let dx = softmax_backward(output, grad, logarithmic).expect("Ascend Softmax backward failed");
        if let Some(parent) = ops.parents[0].as_ref() { grads.register::<Ascend>(parent.id, dx); }
    }
}
impl<C: CheckpointStrategy> SoftmaxBackend for Autodiff<Ascend,C> {
    fn normalized_exponential(input: FloatTensor<Self>, logarithmic: bool) -> Result<FloatTensor<Self>> {
        let output = softmax_forward(input.primitive, logarithmic)?;
        Ok(match SoftmaxBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((output.clone(),logarithmic), output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        })
    }
}

fn reduction_forward(input: Primitive, mean: bool) -> Result<Primitive> {
    let client=input.client.clone();let device=input.device.clone();
    let b=if mean {AscendRuntime::mean_last(&client,buffer(input))?}
        else {AscendRuntime::sum_last(&client,buffer(input))?};
    Ok(Primitive::new(client,b.handle,Metadata::new(b.shape,b.strides),device,b.dtype))
}

impl ReductionBackend for Ascend {
    fn reduce_last(input: FloatTensor<Self>, mean: bool) -> Result<FloatTensor<Self>> {
        reduction_forward(input,mean)
    }
}

#[derive(Clone)]
struct ReductionState {
    shape: Shape,
    client: crate::runtime::ComputeClient<AscendRuntime>,
    device: crate::runtime::AscendDevice,
    mean: bool,
}
impl std::fmt::Debug for ReductionState {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("ReductionState").field("shape",&self.shape).field("device",&self.device)
            .field("mean",&self.mean).finish_non_exhaustive()
    }
}
#[derive(Debug)]
struct ReductionBackward;
impl Backward<Ascend,1> for ReductionBackward {
    type State=ReductionState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let ReductionState {shape,client,device,mean}=ops.state;
        let grad=grads.consume::<Ascend>(&ops.node);
        assert!(grad.device==device && grad.client.same_execution_queue(&client),"Ascend reduction gradient device or queue mismatch");
        let dx=if mean {AscendRuntime::mean_last_backward(&client,shape,buffer(grad))}
            else {AscendRuntime::sum_last_backward(&client,shape,buffer(grad))}
            .expect("Ascend reduction backward failed");
        if let Some(parent)=ops.parents[0].as_ref() {
            grads.register::<Ascend>(parent.id,Primitive::new(client,dx.handle,Metadata::new(dx.shape,dx.strides),device,dx.dtype));
        }
    }
}
impl<C: CheckpointStrategy> ReductionBackend for Autodiff<Ascend,C> {
    fn reduce_last(input:FloatTensor<Self>,mean:bool)->Result<FloatTensor<Self>> {
        let state=ReductionState {shape:input.primitive.meta.shape().clone(),client:input.primitive.client.clone(),
            device:input.primitive.device.clone(),mean};
        let output=reduction_forward(input.primitive,mean)?;
        Ok(match ReductionBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(state,output),
            OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}
