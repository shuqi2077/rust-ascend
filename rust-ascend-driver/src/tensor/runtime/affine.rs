use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,error,
    normalization::{buffer,check_for,run,column_sum},wide_rows};
use ruda_core::{compiler::Compiler,ir::UIntKind,launch::ExecutionMode,tensor::Shape};
use rust_ascend_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,affine_programs::{self,BiasAddSpec}};
type Client=ComputeClient<AscendRuntime>;
fn layout(input:&TensorBuffer)->Result<(usize,u32)> {
    if !(1..=8).contains(&input.shape.len()) {return Err(error("bias add requires rank 1..8"));}
    wide_rows::layout(input)
}
pub(super) fn forward(client:&Client,input:TensorBuffer,bias:TensorBuffer,residual:Option<TensorBuffer>)->Result<TensorBuffer> {
    let (rows,width)=layout(&input)?;check_for(&bias,&[width as usize],"bias add")?;
    if let Some(value)=&residual {check_for(value,&input.shape,"residual bias add")?;}
    let output=buffer(client,input.shape.clone(),false);let spec=BiasAddSpec {rows:rows as u64,width,residual:residual.is_some()};
    let elements=spec.elements().map_err(error)?;
    if elements!=0 {
        let kernel=AscendCompiler.compile(affine_programs::definition(spec).map_err(error)?,&AscendOptions {
            target:Some(AscendTarget::Ascend950DT),elements,..Default::default()
        },ExecutionMode::Checked,UIntKind::U64.into()).map_err(error)?;
        let mut tensors=vec![&input,&bias];if let Some(value)=&residual {tensors.push(value);}tensors.push(&output);
        run(client,kernel,&tensors)?;
    }Ok(output)
}
pub(super) fn bias_backward(client:&Client,grad:TensorBuffer,input_shape:Shape)->Result<TensorBuffer> {
    check_for(&grad,&input_shape,"bias add backward")?;let (rows,width)=layout(&grad)?;
    column_sum(client,grad,rows,width as usize)
}
