use super::{Primitive,Result,SoftmaxBackend,buffer,check_queue,log_softmax};
use crate::{Ascend,Autodiff,driver::CannError,runtime::AscendRuntime};
pub use crate::runtime::{LossReduction,NllLossOptions};
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
    grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::Metadata;
use ruda_tensor::{Backend,TensorPrimitive,api::{Tensor,Int},tensor::{FloatTensor,IntTensor}};

/// Explicit input-only loss autodiff; class weights are fixed inner-backend tensors.
pub trait NllLossBackend:Backend<Device=<Ascend as Backend>::Device> {
    fn nll_loss(input:FloatTensor<Self>,target:IntTensor<Self>,weight:FloatTensor<Ascend>,
        options:NllLossOptions)->Result<FloatTensor<Self>>;
}
fn unquantized<B:Backend>(input:TensorPrimitive<B>)->Result<FloatTensor<B>> {
    match input {TensorPrimitive::Float(input)=>Ok(input),TensorPrimitive::QFloat(_)=>
        Err(CannError::InvalidTensor("NLLLoss requires unquantized FP32 log-probabilities and class weights".into()))}
}
/// Log-probabilities[N,C], integer labels[N], fixed class weights[C].
/// None returns [N]; Mean/Sum return [1]. Labels are classes or the explicit ignore index.
pub fn nll_loss<B:NllLossBackend>(input:Tensor<B,2>,target:Tensor<B,1,Int>,weight:Tensor<Ascend,1>,
    options:NllLossOptions)->Result<Tensor<B,1>> {
    <B as NllLossBackend>::nll_loss(unquantized(input.into_primitive())?,target.into_primitive(),
        unquantized(weight.into_primitive())?,options).map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
/// Cross entropy via native FP32 LogSoftmax and ACLNN NLLLoss, without class weighting.
/// The native LogSoftmax requires positive C divisible by 32; labels are not shifted implicitly.
pub fn cross_entropy<B:NllLossBackend+SoftmaxBackend>(logits:Tensor<B,2>,target:Tensor<B,1,Int>,
    options:NllLossOptions)->Result<Tensor<B,1>> {
    let classes=logits.dims()[1];let device=logits.device();
    let weight=Tensor::<Ascend,1>::ones([classes],(&device,crate::tensor::DType::F32));
    weighted_cross_entropy(logits,target,weight,options)
}
/// Weighted cross entropy; fixed weights do not acquire an autodiff parent.
/// Weighted Mean uses saved non-ignored target weights, not an unmasked batch-size divisor.
pub fn weighted_cross_entropy<B:NllLossBackend+SoftmaxBackend>(logits:Tensor<B,2>,target:Tensor<B,1,Int>,
    weight:Tensor<Ascend,1>,options:NllLossOptions)->Result<Tensor<B,1>> {
    let [rows,classes]=logits.dims();let device=logits.device();
    if target.dims()!=[rows] || weight.dims()!=[classes] || target.device()!=device || weight.device()!=device
        || logits.dtype()!=crate::tensor::DType::F32 || weight.dtype()!=crate::tensor::DType::F32
        || !matches!(target.dtype(),crate::tensor::DType::I32|crate::tensor::DType::I64) {
        return Err(CannError::InvalidTensor("cross entropy requires same-device FP32 logits[N,C], integer labels[N] and FP32 weights[C]".into()));
    }
    nll_loss(log_softmax(logits)?,target,weight,options)
}
fn validate(input:&Primitive,target:&Primitive,weight:&Primitive)->Result<()> {
    use crate::tensor::DType;
    check_queue(input,&[target,weight],"NLLLoss")?;
    if input.dtype!=DType::F32 || weight.dtype!=DType::F32 {
        return Err(CannError::InvalidTensor("RUDA NLLLoss autodiff requires FP32 log-probabilities and weights".into()));
    }
    Ok(())
}
fn primitive(client:&crate::runtime::ComputeClient<AscendRuntime>,device:&<Ascend as Backend>::Device,
    out:crate::runtime::TensorBuffer)->Primitive {
    Primitive::new(client.clone(),out.handle,Metadata::new(out.shape,out.strides),device.clone(),out.dtype)
}
fn forward(input:Primitive,target:Primitive,weight:Primitive,options:NllLossOptions)->Result<Primitive> {
    validate(&input,&target,&weight)?;let client=input.client.clone();let device=input.device.clone();
    let [out,_]=AscendRuntime::nll_loss(&client,buffer(input),buffer(target),buffer(weight),options)?;
    Ok(primitive(&client,&device,out))
}
impl NllLossBackend for Ascend {
    fn nll_loss(input:FloatTensor<Self>,target:IntTensor<Self>,weight:FloatTensor<Ascend>,options:NllLossOptions)
        ->Result<FloatTensor<Self>> {forward(input,target,weight,options)}
}
#[derive(Debug)]
struct NllBackward;
impl Backward<Ascend,1> for NllBackward {
    type State=(Primitive,Primitive,Primitive,Primitive,NllLossOptions);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (input,target,weight,total,options)=ops.state;let grad=grads.consume::<Ascend>(&ops.node);
        check_queue(&grad,&[&input,&target,&weight,&total],"NLLLoss backward").expect("Ascend NLLLoss gradient queue mismatch");
        let client=grad.client.clone();let device=grad.device.clone();
        let dx=AscendRuntime::nll_loss_backward(&client,buffer(grad),buffer(input),buffer(target),buffer(weight),buffer(total),options)
            .expect("Ascend NLLLoss backward failed");
        if let Some(parent)=ops.parents[0].as_ref() {grads.register::<Ascend>(parent.id,primitive(&client,&device,dx));}
    }
}
impl<C:CheckpointStrategy> NllLossBackend for Autodiff<Ascend,C> {
    fn nll_loss(input:FloatTensor<Self>,target:IntTensor<Self>,weight:FloatTensor<Ascend>,options:NllLossOptions)
        ->Result<FloatTensor<Self>> {
        validate(&input.primitive,&target,&weight)?;
        Ok(match NllBackward.prepare::<C>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>{
                let client=input.primitive.client.clone();let device=input.primitive.device.clone();
                let [out,total,target,weight]=AscendRuntime::nll_loss_with_saved_inputs(&client,
                    buffer(input.primitive.clone()),buffer(target),buffer(weight),options)?;
                prep.finish((input.primitive,primitive(&client,&device,target),primitive(&client,&device,weight),
                    primitive(&client,&device,total),options),primitive(&client,&device,out))
            },
            OpsKind::UnTracked(prep)=>prep.finish(forward(input.primitive,target,weight,options)?),
        })
    }
}
