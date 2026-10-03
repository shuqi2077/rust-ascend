//! Native LogSoftmax + ACLNN NLLLoss, with RUDA input gradients and fixed class weights.
use rust_ascend::{Ascend,Autodiff,nn,runtime::{AscendRuntime,RuntimeOptions},
    tensor::{DType,TensorData,api::{Tensor,Int}}};
type AD=Autodiff<Ascend>;
fn close(actual:&[f32],expected:&[f64],name:&str)->Result<(),Box<dyn std::error::Error>> {
    if actual.len()!=expected.len() {return Err(format!("{name}: length mismatch").into());}
    for (i,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs()>3e-5+2e-4*b.abs() {return Err(format!("{name}[{i}]: {a} != {b}").into());}
    }
    Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let toolkit=std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options=RuntimeOptions::new(toolkit);
    if let Some(path)=std::env::var_os("RUDA_CANN_LIBRARY") {options.acl_library=path;}
    if let Some(paths)=std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries=std::env::split_paths(&paths).map(|p|p.into_os_string()).collect();
    }
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device=unsafe {AscendRuntime::initialize_exclusive(options)?};let rows=5;
    for classes in [32,96,4128] {for reduction in [nn::LossReduction::None,nn::LossReduction::Mean,nn::LossReduction::Sum] {
        for ignore_index in [None,Some(-100)] {for weighted in [false,true] {
            let x:Vec<f32>=(0..rows*classes).map(|i|(i%23) as f32/8.-1000.).collect();
            let mut labels=vec![0i64,31,7,0,2];if let Some(ignore)=ignore_index {labels[1]=ignore;}
            let weights:Vec<f32>=(0..classes).map(|c|if weighted {0.5+(c%7) as f32/4.} else {1.}).collect();
            let logits=Tensor::<AD,2>::from_data(TensorData::new(x.clone(),[rows,classes]),(&device,DType::F32)).require_grad();
            let labels_tensor=Tensor::<AD,1,Int>::from_data(TensorData::new(labels.clone(),[rows]),(&device,DType::I64));
            let settings=nn::NllLossOptions {reduction,ignore_index};
            let loss=if weighted {
                let weight=Tensor::<Ascend,1>::from_data(TensorData::new(weights.clone(),[classes]),(&device,DType::F32));
                nn::weighted_cross_entropy(logits.clone(),labels_tensor,weight,settings)?
            } else {nn::cross_entropy(logits.clone(),labels_tensor,settings)?};
            let loss_rows=if reduction==nn::LossReduction::None {rows} else {1};
            let upstream:Vec<f32>=(0..loss_rows).map(|n|1.25+n as f32/4.).collect();
            let dy=Tensor::<AD,1>::from_data(TensorData::new(upstream.clone(),[loss_rows]),(&device,DType::F32));
            let observed=loss.clone().into_data();let gradients=(loss.clone()*dy.clone()+loss*dy).backward();
            let actual_dx=logits.grad(&gradients).ok_or("missing cross entropy logit gradient")?.into_data();
            let total_weight=labels.iter().filter(|&&label|Some(label)!=ignore_index).map(|&label|weights[label as usize] as f64).sum::<f64>();
            let mut row_loss=vec![0.;rows];let mut dx=vec![0.;x.len()];
            for row in 0..rows {
                let label=labels[row];if Some(label)==ignore_index {continue;}
                let values=&x[row*classes..(row+1)*classes];let max=values.iter().map(|&v|v as f64).fold(f64::NEG_INFINITY,f64::max);
                let exp:Vec<_>=values.iter().map(|&v|(v as f64-max).exp()).collect();let sum=exp.iter().sum::<f64>();
                let w=weights[label as usize] as f64;row_loss[row]=w*(sum.ln()+max-values[label as usize] as f64);
                let multiplier=2.*upstream[if reduction==nn::LossReduction::None {row} else {0}] as f64*w
                    /if reduction==nn::LossReduction::Mean {total_weight} else {1.};
                for c in 0..classes {dx[row*classes+c]=multiplier*(exp[c]/sum-if c==label as usize {1.} else {0.});}
            }
            let expected=match reduction {nn::LossReduction::None=>row_loss,
                nn::LossReduction::Mean=>vec![row_loss.iter().sum::<f64>()/total_weight],nn::LossReduction::Sum=>vec![row_loss.iter().sum::<f64>()]};
            close(observed.as_slice::<f32>()?,&expected,"loss")?;close(actual_dx.as_slice::<f32>()?,&dx,"logit gradient")?;
            println!("ASCEND_CROSS_ENTROPY_TENSOR_CASE classes={classes} reduction={reduction:?} ignore={ignore_index:?} weighted={weighted} passed=true");
        }}
    }}
    // Direct NLLLoss has no native Softmax width alignment requirement.
    let input=Tensor::<Ascend,2>::from_data([[-1.0f32,-2.,-3.],[-4.,-5.,-6.]],(&device,DType::F32));
    let target=Tensor::<Ascend,1,Int>::from_data([0i32,2],(&device,DType::I32));
    let weight=Tensor::<Ascend,1>::from_data([1.0f32,2.,3.],(&device,DType::F32));
    let loss=nn::nll_loss(input,target,weight,nn::NllLossOptions {reduction:nn::LossReduction::Mean,ignore_index:None})?;
    close(loss.into_data().as_slice::<f32>()?,&[19./4.],"plain NLLLoss")?;
    println!("ASCEND_CROSS_ENTROPY_TENSOR_DEVICE_OK cases=37");Ok(())
}
