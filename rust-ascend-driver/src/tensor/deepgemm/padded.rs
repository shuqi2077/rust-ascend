//! Logical matrix dimensions with explicitly zero-padded native BF16 buffers.
use super::{GemmSpec,Transpose};
use crate::{CannError,tensor::{DType,TensorLayout,layout::invalid}};

#[derive(Debug,Clone,PartialEq,Eq)]
pub struct PaddedGemmSpec {
    pub a:TensorLayout,
    pub b:TensorLayout,
    pub output:TensorLayout,
    pub native:GemmSpec,
}
impl PaddedGemmSpec {
    /// Matching rank-2 or rank-3 matrices; no batch broadcast or empty matrix axes.
    /// Physical M/N/K are rounded up to 16, within the original native domain.
    pub fn new(a:&TensorLayout,b:&TensorLayout,ta:Transpose,tb:Transpose)->Result<Self,CannError> {
        let rank=a.shape().len();
        if !(rank==2 || rank==3) || b.shape().len()!=rank
            || !matches!(a.dtype(),DType::F32|DType::BF16) || !matches!(b.dtype(),DType::F32|DType::BF16) {
            return Err(invalid("padded GEMM requires matching rank-2/3 FP32/BF16 inputs"));
        }
        if rank==3 && a.shape()[0]!=b.shape()[0] {return Err(invalid("padded GEMM does not broadcast batches"));}
        let (m,k)=if ta==Transpose::No {(a.shape()[rank-2],a.shape()[rank-1])} else {(a.shape()[rank-1],a.shape()[rank-2])};
        let (kb,n)=if tb==Transpose::No {(b.shape()[rank-2],b.shape()[rank-1])} else {(b.shape()[rank-1],b.shape()[rank-2])};
        if k!=kb {return Err(invalid("padded GEMM logical contraction mismatch"));}
        let pad=|layout:&TensorLayout|->Result<TensorLayout,CannError> {
            let mut shape=layout.shape().to_vec();
            for size in &mut shape[rank-2..] {
                if *size<=0 {return Err(invalid("padded GEMM matrix axes must be positive"));}
                *size=size.checked_add(15).map(|v|v/16*16).filter(|&v|v<=i32::MAX as i64)
                    .ok_or_else(||invalid("padded GEMM aligned axis exceeds INT32_MAX"))?;
            }
            TensorLayout::contiguous(&shape,DType::BF16)
        };
        let native=GemmSpec::new(&pad(a)?,&pad(b)?,ta,tb,DType::F32)?;
        let output=TensorLayout::contiguous(&if rank==3 {vec![a.shape()[0],m,n]} else {vec![m,n]},DType::F32)?;
        Ok(Self {a:a.clone(),b:b.clone(),output,native})
    }
    /// Native derivatives of the padded graph. Each result must be cropped to its
    /// original physical input shape, not to the logical op(A)/op(B) shape.
    pub fn backward_specs(&self)->Result<[GemmSpec;2],CannError> {
        let dy=TensorLayout::contiguous(self.native.output_layout().shape(),DType::BF16)?;
        let opposite=|t|if t==Transpose::No {Transpose::Yes} else {Transpose::No};
        let ta=self.native.ta;let tb=self.native.tb;
        let da=if ta==Transpose::No {GemmSpec::new(&dy,&self.native.b,Transpose::No,opposite(tb),DType::F32)?}
            else {GemmSpec::new(&self.native.b,&dy,tb,Transpose::Yes,DType::F32)?};
        let db=if tb==Transpose::No {GemmSpec::new(&self.native.a,&dy,opposite(ta),Transpose::No,DType::F32)?}
            else {GemmSpec::new(&dy,&self.native.a,Transpose::Yes,ta,DType::F32)?};
        Ok([da,db])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_tail_and_aligned_matrices_preserve_logical_shape_and_transposed_gradients() {
        for batch in [None,Some(2)] {for (m,n,k) in [(1,1,1),(3,17,7),(17,33,65),(16,32,48)] {
            for ta in [Transpose::No,Transpose::Yes] {for tb in [Transpose::No,Transpose::Yes] {
                let mut a=if ta==Transpose::No {vec![m,k]} else {vec![k,m]};
                let mut b=if tb==Transpose::No {vec![k,n]} else {vec![n,k]};
                let mut out=vec![m,n];if let Some(batch)=batch {a.insert(0,batch);b.insert(0,batch);out.insert(0,batch);}
                let a=TensorLayout::contiguous(&a,DType::F32).unwrap();let b=TensorLayout::contiguous(&b,DType::BF16).unwrap();
                let spec=PaddedGemmSpec::new(&a,&b,ta,tb).unwrap();assert_eq!(spec.output.shape(),out);
                assert_eq!(spec.native.dimensions(),[(m+15) as u32/16*16,(n+15) as u32/16*16,(k+15) as u32/16*16]);
                let [da,db]=spec.backward_specs().unwrap();assert_eq!(da.output_layout().shape(),spec.native.a.shape());
                assert_eq!(db.output_layout().shape(),spec.native.b.shape());assert_eq!(da.output_layout().dtype(),DType::F32);
                assert_eq!(spec.native.a.dtype(),DType::BF16);assert_eq!(spec.native.b.dtype(),DType::BF16);
            }}
        }}
    }
    #[test]
    fn padding_does_not_disguise_contract_batch_dtype_or_native_domain_errors() {
        let t=|shape:&[i64],dtype|TensorLayout::contiguous(shape,dtype).unwrap();
        for (a,b) in [(t(&[3,7],DType::F32),t(&[8,17],DType::F32)),
            (t(&[0,7],DType::F32),t(&[7,17],DType::F32)),(t(&[2,3,7],DType::F32),t(&[1,7,17],DType::F32)),
            (t(&[3,7],DType::F16),t(&[7,17],DType::F32)),(t(&[i32::MAX as i64,7],DType::F32),t(&[7,17],DType::F32)),
            (t(&[4097,3,7],DType::F32),t(&[4097,7,17],DType::F32))] {
            assert!(PaddedGemmSpec::new(&a,&b,Transpose::No,Transpose::No).is_err());
        }
    }
}
