//! Native Ascend operations on RUDA tensors and RUDA's existing autodiff graph.
mod attention;
mod affine;
pub use affine::{BiasAddBackend,bias_add,residual_bias_add};
mod piecewise;
pub use piecewise::{PiecewiseBackend,PiecewiseActivation,relu,clamp,leaky_relu,hard_sigmoid,hard_swish};
mod padded_attention;
pub use padded_attention::{scaled_dot_product_attention_padded_bf16_fp32,causal_attention_padded_bf16_fp32,
    grouped_query_attention_padded_bf16_fp32,causal_grouped_query_attention_padded_bf16_fp32};
mod rotary;
mod feed_forward;
mod gelu_ffn;
pub use gelu_ffn::{GeluMode,gelu_mlp_padded_bf16_fp32_nd,geglu_padded_bf16_fp32_nd,
    gelu_mlp_frozen_padded_bf16_fp32_nd,geglu_frozen_padded_bf16_fp32_nd};
mod embedding;
mod loss;
mod heads;
mod frozen;
mod padded;
mod projection;
pub use projection::{linear_padded_bf16_fp32_nd,linear_frozen_padded_bf16_fp32_nd,
    lora_padded_linear_bf16_fp32_nd,lora_frozen_padded_linear_bf16_fp32_nd,
    swiglu_padded_bf16_fp32_nd,swiglu_frozen_padded_bf16_fp32_nd};
pub use padded::{PaddedMatmulBf16Fp32Backend,FrozenPaddedLinearBf16Fp32Backend,
    matmul_padded_bf16_fp32,linear_padded_bf16_fp32,lora_padded_linear_bf16_fp32,swiglu_padded_bf16_fp32,
    linear_frozen_padded_bf16_fp32,lora_frozen_padded_linear_bf16_fp32,swiglu_frozen_padded_bf16_fp32};
pub use frozen::{FrozenLinearBf16Fp32Backend,linear_frozen_bf16_fp32,lora_frozen_linear_bf16_fp32,swiglu_frozen_bf16_fp32,
    FrozenEmbeddingBf16Fp32Backend,embedding_frozen_bf16_fp32,embedding_frozen_bf16_fp32_nd};
pub use heads::{RepeatKvBackend,repeat_kv_heads};
pub use attention::{scaled_dot_product_attention_bf16_fp32,causal_attention_bf16_fp32,
    grouped_query_attention_bf16_fp32,causal_grouped_query_attention_bf16_fp32};
mod mask;
pub use mask::{CausalMaskBackend,causal_mask};
pub use rotary::{RotaryBackend,rotary,RotaryPrefixBackend,rotary_prefix};
pub use feed_forward::{lora_linear_bf16_fp32,swiglu_bf16_fp32};
pub use embedding::{EmbeddingBackend,EmbeddingOptions,embedding,embedding_nd};
pub use loss::{NllLossBackend,LossReduction,NllLossOptions,nll_loss,cross_entropy,weighted_cross_entropy};
pub use crate::runtime::RotaryLayout;
use crate::{Ascend, Autodiff, driver::CannError, runtime::{AscendRuntime, TensorBuffer, Transpose}};
use ruda_autodiff::{checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients, ops::{Backward, Ops, OpsKind}};
use ruda_core::tensor::{Metadata, Shape};
use ruda_tensor::{Backend, TensorPrimitive, api::Tensor, tensor::FloatTensor};
use ruda_tensor_device::RudaTensor;

type Primitive = RudaTensor<AscendRuntime>;
type Result<T> = std::result::Result<T, CannError>;

/// Backend extension for explicitly selected BF16-compute Dense/Batched matmul.
pub trait MatmulBf16Fp32Backend: Backend {
    fn matmul_bf16_fp32(a:FloatTensor<Self>,b:FloatTensor<Self>,ta:Transpose,tb:Transpose)->Result<FloatTensor<Self>>;
}

/// Explicit BF16-compute matmul of contiguous FP32 rank-2 or matching-batch rank-3 tensors.
/// Output and gradients are FP32; forward inputs and upstream gradients are cast to BF16.
/// All four transpose combinations are supported without materializing transposed copies.
pub fn matmul_bf16_fp32<B:MatmulBf16Fp32Backend,const D:usize>(a:Tensor<B,D>,b:Tensor<B,D>,
    ta:Transpose,tb:Transpose)->Result<Tensor<B,D>> {
    let unquantized=|primitive:TensorPrimitive<B>|match primitive {
        TensorPrimitive::Float(tensor)=>Ok(tensor),
        TensorPrimitive::QFloat(_)=>Err(CannError::InvalidTensor("BF16-compute matmul does not dequantize inputs implicitly".into())),
    };
    B::matmul_bf16_fp32(unquantized(a.into_primitive())?,unquantized(b.into_primitive())?,ta,tb)
        .map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Backend extension for explicitly selected BF16-compute, FP32-storage linear operations.
pub trait LinearBf16Fp32Backend: Backend {
    fn linear_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Self>)->Result<FloatTensor<Self>>;
}

/// Y = X W^T for contiguous FP32 X[M,K] and W[N,K], with M/N/K positive multiples of 16.
/// Casts X/W and upstream derivatives to BF16 on-device; output and both gradients are FP32.
/// This explicit compute mode does not change generic tensor matmul or add bias/broadcasting.
pub fn linear_bf16_fp32<B:LinearBf16Fp32Backend>(input:Tensor<B,2>,weight:Tensor<B,2>)
    ->Result<Tensor<B,2>> {
    let unquantized=|primitive:TensorPrimitive<B>|match primitive {
        TensorPrimitive::Float(tensor)=>Ok(tensor),
        TensorPrimitive::QFloat(_)=>Err(CannError::InvalidTensor("BF16-compute linear does not dequantize inputs implicitly".into())),
    };
    B::linear_bf16_fp32(unquantized(input.into_primitive())?,unquantized(weight.into_primitive())?)
        .map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
}

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

/// Backend extension for native fused SiLU(gate) * up and both input gradients.
pub trait SiluMulBackend: Backend {
    fn silu_mul(gate:FloatTensor<Self>,up:FloatTensor<Self>)->Result<FloatTensor<Self>>;
}

/// Native FP32 SiLU(gate) * up on equal-shaped contiguous RUDA tensors.
pub fn silu_mul<B:SiluMulBackend,const D:usize>(gate:Tensor<B,D>,up:Tensor<B,D>)->Result<Tensor<B,D>> {
    let unquantized=|primitive:TensorPrimitive<B>|match primitive {
        TensorPrimitive::Float(tensor)=>Ok(tensor),
        TensorPrimitive::QFloat(_)=>Err(CannError::InvalidTensor("native SiLU Mul does not dequantize inputs implicitly".into())),
    };
    B::silu_mul(unquantized(gate.into_primitive())?,unquantized(up.into_primitive())?)
        .map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
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
/// Positive unaligned widths and widths above 4096 use tiled native passes.
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
/// The shared weight has rank one; the logical width must be positive.
/// Wider rows use device-side tiled statistics and the forward's saved reciprocal RMS.
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

fn silu_mul_forward(gate:Primitive,up:Primitive)->Result<Primitive> {
    check_queue(&gate,&[&up],"SiLU Mul")?;
    let client=gate.client.clone();let device=gate.device.clone();
    let b=AscendRuntime::silu_mul(&client,buffer(gate),buffer(up))?;
    Ok(Primitive::new(client,b.handle,Metadata::new(b.shape,b.strides),device,b.dtype))
}
fn silu_mul_backward(gate:Primitive,up:Primitive,grad:Primitive)->Result<[Primitive;2]> {
    check_queue(&gate,&[&up,&grad],"SiLU Mul backward")?;
    let client=gate.client.clone();let device=gate.device.clone();
    let result=AscendRuntime::silu_mul_backward(&client,buffer(gate),buffer(up),buffer(grad))?;
    Ok(result.map(|b|Primitive::new(client.clone(),b.handle,Metadata::new(b.shape,b.strides),device.clone(),b.dtype)))
}
impl SiluMulBackend for Ascend {
    fn silu_mul(gate:FloatTensor<Self>,up:FloatTensor<Self>)->Result<FloatTensor<Self>> {
        silu_mul_forward(gate,up)
    }
}
#[derive(Debug)]
struct SiluMulBackward;
impl Backward<Ascend,2> for SiluMulBackward {
    type State=(Primitive,Primitive);
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (gate,up)=ops.state;
        let grad=grads.consume::<Ascend>(&ops.node);
        let derivatives=silu_mul_backward(gate,up,grad).expect("Ascend SiLU Mul backward failed");
        for (parent,gradient) in ops.parents.into_iter().zip(derivatives) {
            if let Some(parent)=parent {grads.register::<Ascend>(parent.id,gradient);}
        }
    }
}
impl<C:CheckpointStrategy> SiluMulBackend for Autodiff<Ascend,C> {
    fn silu_mul(gate:FloatTensor<Self>,up:FloatTensor<Self>)->Result<FloatTensor<Self>> {
        let x=gate.primitive;let u=up.primitive;
        let output=silu_mul_forward(x.clone(),u.clone())?;
        Ok(match SiluMulBackward.prepare::<C>([gate.node,up.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((x,u),output),
            OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}

fn linear_forward(input:Primitive,weight:Primitive)->Result<[Primitive;3]> {
    check_queue(&input,&[&weight],"BF16-compute linear")?;
    let client=input.client.clone();let device=input.device.clone();
    let result=AscendRuntime::linear_bf16_fp32(&client,buffer(input),buffer(weight))?;
    Ok(result.map(|b|Primitive::new(client.clone(),b.handle,Metadata::new(b.shape,b.strides),device.clone(),b.dtype)))
}
fn linear_backward(input:Primitive,weight:Primitive,grad:Primitive)->Result<[Primitive;2]> {
    check_queue(&input,&[&weight,&grad],"BF16-compute linear backward")?;
    let client=input.client.clone();let device=input.device.clone();
    let result=AscendRuntime::linear_bf16_fp32_backward(&client,buffer(input),buffer(weight),buffer(grad))?;
    Ok(result.map(|b|Primitive::new(client.clone(),b.handle,Metadata::new(b.shape,b.strides),device.clone(),b.dtype)))
}
impl LinearBf16Fp32Backend for Ascend {
    fn linear_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Self>)->Result<FloatTensor<Self>> {
        linear_forward(input,weight).map(|[output,_,_]|output)
    }
}
#[derive(Debug)]
struct LinearBf16Fp32Backward;
impl Backward<Ascend,2> for LinearBf16Fp32Backward {
    type State=(Primitive,Primitive);
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (input,weight)=ops.state;
        let grad=grads.consume::<Ascend>(&ops.node);
        let derivatives=linear_backward(input,weight,grad).expect("Ascend BF16-compute linear backward failed");
        for (parent,gradient) in ops.parents.into_iter().zip(derivatives) {
            if let Some(parent)=parent {grads.register::<Ascend>(parent.id,gradient);}
        }
    }
}
impl<C:CheckpointStrategy> LinearBf16Fp32Backend for Autodiff<Ascend,C> {
    fn linear_bf16_fp32(input:FloatTensor<Self>,weight:FloatTensor<Self>)->Result<FloatTensor<Self>> {
        let [output,x,w]=linear_forward(input.primitive,weight.primitive)?;
        Ok(match LinearBf16Fp32Backward.prepare::<C>([input.node,weight.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((x,w),output),
            OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}

fn matmul_forward(a:Primitive,b:Primitive,ta:Transpose,tb:Transpose)->Result<[Primitive;3]> {
    check_queue(&a,&[&b],"BF16-compute matmul")?;
    let client=a.client.clone();let device=a.device.clone();
    let result=AscendRuntime::gemm_bf16_fp32(&client,buffer(a),buffer(b),ta,tb)?;
    Ok(result.map(|b|Primitive::new(client.clone(),b.handle,Metadata::new(b.shape,b.strides),device.clone(),b.dtype)))
}
fn matmul_backward(a:Primitive,b:Primitive,grad:Primitive,ta:Transpose,tb:Transpose)->Result<[Primitive;2]> {
    check_queue(&a,&[&b,&grad],"BF16-compute matmul backward")?;
    let client=a.client.clone();let device=a.device.clone();
    let result=AscendRuntime::gemm_bf16_fp32_backward(&client,buffer(a),buffer(b),buffer(grad),ta,tb)?;
    Ok(result.map(|b|Primitive::new(client.clone(),b.handle,Metadata::new(b.shape,b.strides),device.clone(),b.dtype)))
}
impl MatmulBf16Fp32Backend for Ascend {
    fn matmul_bf16_fp32(a:FloatTensor<Self>,b:FloatTensor<Self>,ta:Transpose,tb:Transpose)->Result<FloatTensor<Self>> {
        matmul_forward(a,b,ta,tb).map(|[output,_,_]|output)
    }
}
#[derive(Debug)]
struct MatmulBf16Fp32Backward;
impl Backward<Ascend,2> for MatmulBf16Fp32Backward {
    type State=(Primitive,Primitive,Transpose,Transpose);
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (a,b,ta,tb)=ops.state;
        let grad=grads.consume::<Ascend>(&ops.node);
        let derivatives=matmul_backward(a,b,grad,ta,tb).expect("Ascend BF16-compute matmul backward failed");
        for (parent,gradient) in ops.parents.into_iter().zip(derivatives) {
            if let Some(parent)=parent {grads.register::<Ascend>(parent.id,gradient);}
        }
    }
}
impl<C:CheckpointStrategy> MatmulBf16Fp32Backend for Autodiff<Ascend,C> {
    fn matmul_bf16_fp32(a:FloatTensor<Self>,b:FloatTensor<Self>,ta:Transpose,tb:Transpose)->Result<FloatTensor<Self>> {
        let [output,a_saved,b_saved]=matmul_forward(a.primitive,b.primitive,ta,tb)?;
        Ok(match MatmulBf16Fp32Backward.prepare::<C>([a.node,b.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((a_saved,b_saved,ta,tb),output),
            OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
}
