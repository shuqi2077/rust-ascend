use super::{AscendRuntime,ComputeClient,Result,TensorBuffer,CausalMaskSpec,error,indexing::allocate,normalization::run};
use crate::tensor::{TensorLayout,DType};
use ruda_core::{compiler::Compiler,ir::UIntKind,launch::ExecutionMode};
use rust_ascend_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,mask_programs};
pub(super) fn causal(client:&ComputeClient<AscendRuntime>,spec:CausalMaskSpec)->Result<TensorBuffer> {
    let n=spec.elements().map_err(error)?;
    let layout=TensorLayout::contiguous(&[spec.batch as i64,spec.queries as i64,spec.keys as i64],DType::F32)?;
    let out=allocate(client,&layout);
    if n!=0 {
        let kernel=AscendCompiler.compile(mask_programs::definition(spec).map_err(error)?,&AscendOptions {target:Some(AscendTarget::Ascend950DT),elements:n,..Default::default()},
            ExecutionMode::Checked,UIntKind::U64.into()).map_err(error)?;
        run(client,kernel,&[&out])?;
    }
    Ok(out)
}
