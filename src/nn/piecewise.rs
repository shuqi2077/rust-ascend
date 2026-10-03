use super::{Primitive,Result,buffer};
use crate::{Ascend,Autodiff,driver::CannError,runtime::{AscendRuntime,AscendDevice,ComputeClient,TensorBuffer}};
pub use crate::runtime::PiecewiseActivation;
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::Metadata;
use ruda_tensor::{Backend,TensorPrimitive,api::Tensor,tensor::FloatTensor};

pub trait PiecewiseBackend:Backend {
    fn piecewise_activation(input:FloatTensor<Self>,activation:PiecewiseActivation)->Result<FloatTensor<Self>>;
}
fn apply<B:PiecewiseBackend,const D:usize>(input:Tensor<B,D>,activation:PiecewiseActivation)->Result<Tensor<B,D>> {
    let input=match input.into_primitive() {
        TensorPrimitive::Float(t)=>t,TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("piecewise activation requires unquantized FP32".into())),
    };
    B::piecewise_activation(input,activation).map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
/// Explicit native FP32 ReLU: X <= 0 becomes +0; unordered NaN lanes pass through.
/// Accepts contiguous rank 1..8, total count <=u32, including empty axes.
pub fn relu<B:PiecewiseBackend,const D:usize>(input:Tensor<B,D>)->Result<Tensor<B,D>> {
    apply(input,PiecewiseActivation::Relu)
}
/// Explicit FP32 LeakyReLU: only X < 0 is multiplied by the supplied slope.
/// Backward retains the original RUDA masked branches and shared-parent addition.
pub fn leaky_relu<B:PiecewiseBackend,const D:usize>(input:Tensor<B,D>,negative_slope:f32)->Result<Tensor<B,D>> {
    apply(input,PiecewiseActivation::LeakyRelu {negative_slope})
}
/// Explicit native FP32 upper-then-lower Clamp, preserving RUDA's default comparison/select order.
/// Bounds may be infinite, unordered or NaN; no implicit validation, normalization or bound swapping.
/// Equal-boundary lanes retain their input and derivative; strictly clipped derivatives are +0.
pub fn clamp<B:PiecewiseBackend,const D:usize>(input:Tensor<B,D>,min:f32,max:f32)->Result<Tensor<B,D>> {
    apply(input,PiecewiseActivation::Clamp {min,max})
}
/// FP32 alpha * X + beta, then explicit upper/lower Clamp to [0,1].
/// Reuses native arithmetic and RUDA autodiff; derivatives pass through equal boundaries.
pub fn hard_sigmoid<B:PiecewiseBackend,const D:usize>(input:Tensor<B,D>,alpha:f32,beta:f32)->Result<Tensor<B,D>> {
    apply(input,PiecewiseActivation::HardSigmoid {alpha,beta})
}
/// FP32 X * HardSigmoid(X, 1/6, 1/2), retaining RUDA's shared-input computation graph.
pub fn hard_swish<B:PiecewiseBackend,const D:usize>(input:Tensor<B,D>)->Result<Tensor<B,D>> {
    Ok(input.clone()*hard_sigmoid(input,1f32/6.,0.5)?)
}
fn forward(input:Primitive,activation:PiecewiseActivation)->Result<Primitive> {
    let client=input.client.clone();let device=input.device.clone();
    let out=AscendRuntime::piecewise_activation(&client,buffer(input),activation)?;
    Ok(Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype))
}
impl PiecewiseBackend for Ascend {
    fn piecewise_activation(input:FloatTensor<Self>,activation:PiecewiseActivation)->Result<FloatTensor<Self>> {forward(input,activation)}
}
#[derive(Clone)]
struct PiecewiseState {input:TensorBuffer,activation:PiecewiseActivation,client:ComputeClient<AscendRuntime>,device:AscendDevice}
impl std::fmt::Debug for PiecewiseState {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("PiecewiseState").field("shape",&self.input.shape).field("activation",&self.activation).field("device",&self.device).finish_non_exhaustive()
    }
}
#[derive(Debug)]
struct PiecewiseBackward;
impl Backward<Ascend,1> for PiecewiseBackward {
    type State=PiecewiseState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let PiecewiseState {input,activation,client,device}=ops.state;let grad=grads.consume::<Ascend>(&ops.node);
        assert!(grad.device==device && grad.client.same_execution_queue(&client),"piecewise gradient device/queue mismatch");
        if let Some(parent)=ops.parents[0].as_ref() {
            let out=AscendRuntime::piecewise_activation_backward(&client,input,buffer(grad),activation).expect("Ascend piecewise gradient failed");
            grads.register::<Ascend>(parent.id,Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype));
        }
    }
}
impl<C:CheckpointStrategy> PiecewiseBackend for Autodiff<Ascend,C> {
    fn piecewise_activation(input:FloatTensor<Self>,activation:PiecewiseActivation)->Result<FloatTensor<Self>> {
        let output=forward(input.primitive.clone(),activation)?;
        Ok(match PiecewiseBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>{
                let client=input.primitive.client.clone();let device=input.primitive.device.clone();
                // A tracked graph owns an independent on-device input snapshot, not a mutable alias.
                let saved=AscendRuntime::copy_contiguous(&client,buffer(input.primitive))?;
                prep.finish(PiecewiseState {input:saved,activation,client,device},output)
            },OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}
