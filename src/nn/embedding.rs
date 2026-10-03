use super::{Primitive,Result,buffer,check_queue};
use crate::{Ascend,Autodiff,driver::CannError,runtime::AscendRuntime};
pub use crate::runtime::EmbeddingOptions;
use ruda_autodiff::{checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
    grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_core::tensor::Metadata;
use ruda_tensor::{Backend,TensorPrimitive,api::{Tensor,Int},tensor::{FloatTensor,IntTensor}};

/// Explicit embedding lookup and dense weight autodiff on integer device IDs.
pub trait EmbeddingBackend:Backend {
    fn embedding(weight:FloatTensor<Self>,indices:IntTensor<Self>,options:EmbeddingOptions)
        ->Result<FloatTensor<Self>>;
}
/// FP32 weight[V,H] and INT32/INT64 IDs[B,S] produce output[B,S,H].
/// Padding affects only the derivative; forward values are read from the supplied table.
pub fn embedding<B:EmbeddingBackend>(weight:Tensor<B,2>,indices:Tensor<B,2,Int>,
    options:EmbeddingOptions)->Result<Tensor<B,3>> {
    embedding_nd::<B,2,3>(weight,indices,options)
}
/// IDs have rank D in 1..7; output rank O must be D+1, appending the table width.
/// IDs must be within [0,V); no host transfer, implicit clamp or floating-ID conversion.
pub fn embedding_nd<B:EmbeddingBackend,const D:usize,const O:usize>(weight:Tensor<B,2>,
    indices:Tensor<B,D,Int>,options:EmbeddingOptions)->Result<Tensor<B,O>> {
    if !(1..=7).contains(&D) || D.checked_add(1)!=Some(O) {
        return Err(CannError::InvalidTensor("embedding output rank must append one axis to rank 1..7 IDs".into()));
    }
    let weight=match weight.into_primitive() {
        TensorPrimitive::Float(weight)=>weight,
        TensorPrimitive::QFloat(_)=>return Err(CannError::InvalidTensor("embedding does not dequantize its table implicitly".into())),
    };
    <B as EmbeddingBackend>::embedding(weight,indices.into_primitive(),options)
        .map(|out|Tensor::from_primitive(TensorPrimitive::Float(out)))
}
fn validate(weight:&Primitive,indices:&Primitive,options:EmbeddingOptions)->Result<()> {
    use crate::tensor::DType;
    check_queue(weight,&[indices],"embedding")?;
    if weight.dtype!=DType::F32 || weight.meta.shape().len()!=2 {
        return Err(CannError::InvalidTensor("RUDA embedding autodiff requires a rank-2 FP32 table".into()));
    }
    if options.padding_idx.is_some_and(|index|index>=weight.meta.shape()[0]) {
        return Err(CannError::InvalidTensor("embedding padding row is outside its table".into()));
    }
    Ok(())
}
fn forward(weight:Primitive,indices:Primitive,options:EmbeddingOptions)->Result<Primitive> {
    validate(&weight,&indices,options)?;
    let client=weight.client.clone();let device=weight.device.clone();
    let out=AscendRuntime::embedding(&client,buffer(weight),buffer(indices))?;
    Ok(Primitive::new(client,out.handle,Metadata::new(out.shape,out.strides),device,out.dtype))
}
impl EmbeddingBackend for Ascend {
    fn embedding(weight:FloatTensor<Self>,indices:IntTensor<Self>,options:EmbeddingOptions)
        ->Result<FloatTensor<Self>> {forward(weight,indices,options)}
}
#[derive(Debug)]
struct EmbeddingBackward;
impl Backward<Ascend,1> for EmbeddingBackward {
    type State=(Primitive,u64,EmbeddingOptions);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (ids,rows,options)=ops.state;
        let grad=grads.consume::<Ascend>(&ops.node);
        check_queue(&grad,&[&ids],"embedding backward").expect("Ascend embedding gradient queue mismatch");
        let client=grad.client.clone();let device=grad.device.clone();
        let dw=AscendRuntime::embedding_backward(&client,buffer(grad),buffer(ids),rows,options)
            .expect("Ascend embedding backward failed");
        if let Some(parent)=ops.parents[0].as_ref() {
            grads.register::<Ascend>(parent.id,Primitive::new(client,dw.handle,
                Metadata::new(dw.shape,dw.strides),device,dw.dtype));
        }
    }
}
impl<C:CheckpointStrategy> EmbeddingBackend for Autodiff<Ascend,C> {
    fn embedding(weight:FloatTensor<Self>,indices:IntTensor<Self>,options:EmbeddingOptions)
        ->Result<FloatTensor<Self>> {
        validate(&weight.primitive,&indices,options)?;
        Ok(match EmbeddingBackward.prepare::<C>([weight.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>{
                let rows=weight.primitive.meta.shape()[0] as u64;
                let client=weight.primitive.client.clone();let device=weight.primitive.device.clone();
                let [out,saved]=AscendRuntime::embedding_with_saved_indices(&client,
                    buffer(weight.primitive),buffer(indices))?;
                let out=Primitive::new(client.clone(),out.handle,Metadata::new(out.shape,out.strides),device.clone(),out.dtype);
                let saved=Primitive::new(client,saved.handle,Metadata::new(saved.shape,saved.strides),device,saved.dtype);
                prep.finish((saved,rows,options),out)
            },
            OpsKind::UnTracked(prep)=>prep.finish(forward(weight.primitive,indices,options)?),
        })
    }
}
