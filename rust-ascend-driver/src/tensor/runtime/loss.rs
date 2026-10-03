use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,WORKER,NllLossOptions,error};
use super::indexing::{layout,allocate,copy_contiguous};
use crate::tensor::{TensorLayout,loss::{forward_layouts,backward_layout}};
type Client=ComputeClient<AscendRuntime>;
fn launch(client:&Client,tensors:Vec<&TensorBuffer>,layouts:Vec<TensorLayout>,options:NllLossOptions,backward:bool)->Result<()> {
    client.flush().map_err(error)?;
    let guards=tensors.into_iter().map(|t|client.get_resource(t.handle.clone()).map_err(error)).collect::<Result<Vec<_>>>()?;
    let resources=guards.iter().zip(&layouts).map(|(guard,layout)| {
        let mut resource=guard.resource().clone();if resource.byte_len()<layout.byte_len() {return Err(error("NLLLoss resource is too short"));}
        resource.size=layout.byte_len();Ok(resource)
    }).collect::<Result<Vec<_>>>()?;
    let worker=&WORKER.get().ok_or_else(||error("Ascend runtime is not initialized"))?.1;
    let result=worker.call(move|state|state.nll_loss(layouts,resources,options,backward));drop(guards);result
}
pub(super) fn forward(client:&Client,input:TensorBuffer,target:TensorBuffer,weight:TensorBuffer,options:NllLossOptions)->Result<[TensorBuffer;2]> {
    let x=layout(&input)?;let t=layout(&target)?;let w=layout(&weight)?;
    let [output,total]=forward_layouts(&x,&t,&w,options)?;
    let out=allocate(client,&output);let tw=allocate(client,&total);
    launch(client,vec![&input,&target,&weight,&out,&tw],vec![x,t,w,output,total],options,false)?;Ok([out,tw])
}
pub(super) fn saved_forward(client:&Client,input:TensorBuffer,target:TensorBuffer,weight:TensorBuffer,options:NllLossOptions)->Result<[TensorBuffer;4]> {
    forward_layouts(&layout(&input)?,&layout(&target)?,&layout(&weight)?,options)?;
    let saved_target=copy_contiguous(client,target)?;let saved_weight=copy_contiguous(client,weight)?;
    let [out,total]=forward(client,input,saved_target.clone(),saved_weight.clone(),options)?;
    Ok([out,total,saved_target,saved_weight])
}
pub(super) fn backward(client:&Client,grad:TensorBuffer,input:TensorBuffer,target:TensorBuffer,weight:TensorBuffer,total_weight:TensorBuffer,
    options:NllLossOptions)->Result<TensorBuffer> {
    let layouts=[layout(&grad)?,layout(&input)?,layout(&target)?,layout(&weight)?,layout(&total_weight)?];
    let output=backward_layout(&layouts[0],&layouts[1],&layouts[2],&layouts[3],&layouts[4],options)?;
    let out=allocate(client,&output);let mut layouts=layouts.to_vec();layouts.push(output);
    launch(client,vec![&grad,&input,&target,&weight,&total_weight,&out],layouts,options,true)?;Ok(out)
}
