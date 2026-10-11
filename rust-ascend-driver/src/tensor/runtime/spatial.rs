use super::{
    AscendResource, AscendRuntime, ComputeClient, Result, TensorBuffer, WORKER,
    descriptor::Descriptor, error, typed, worker::State,
};
use crate::tensor::{DType, TensorLayout, ffi::*};
use ruda_core::tensor::{
    Shape,
    spatial::{
        calculate_conv_output_shape, calculate_conv_transpose_output_shape,
        calculate_pool_output_shape,
    },
};

#[derive(Debug, Clone)]
pub struct SpatialConvOptions<const N: usize> {
    pub stride: [usize; N],
    pub padding: [usize; N],
    pub dilation: [usize; N],
    pub groups: usize,
    pub transposed: bool,
    pub output_padding: [usize; N],
}

#[derive(Debug, Clone)]
pub struct Pool2dOptions {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub ceil_mode: bool,
    pub count_include_pad: bool,
}

fn floating(inputs: &[TensorLayout]) -> Result<()> {
    if inputs.is_empty()
        || !matches!(inputs[0].dtype(), DType::F32 | DType::F16 | DType::BF16)
        || inputs
            .iter()
            .any(|l| l.dtype() != inputs[0].dtype() || !is_contiguous(l))
    {
        return Err(error(
            "spatial operators require matching contiguous F32/F16/BF16 storage",
        ));
    }
    Ok(())
}
fn is_contiguous(value: &TensorLayout) -> bool {
    TensorLayout::contiguous(value.shape(), value.dtype())
        .is_ok_and(|l| l.strides() == value.strides())
}
fn shape(layout: &TensorLayout) -> Shape {
    Shape::from(
        layout
            .shape()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>(),
    )
}
fn layout(shape: Shape, dtype: DType) -> Result<TensorLayout> {
    TensorLayout::contiguous(&ints(&shape)?, dtype)
}
fn ints(values: &[usize]) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|&n| i64::try_from(n).map_err(error))
        .collect()
}
fn conv_layout<const N: usize>(
    x: &TensorLayout,
    w: &TensorLayout,
    options: &SpatialConvOptions<N>,
) -> Result<TensorLayout> {
    floating(&[x.clone(), w.clone()])?;
    if !matches!(N, 2 | 3)
        || x.shape().len() != N + 2
        || w.shape().len() != N + 2
        || options.groups == 0
        || options.groups > 65535
        || options.stride.contains(&0)
        || options.dilation.contains(&0)
        || w.shape()[2..].contains(&0)
    {
        return Err(error(
            "invalid convolution rank, groups, kernel, stride or dilation",
        ));
    }
    let group = options.groups as i64;
    let (input_channels, output_channels) = if options.transposed {
        (w.shape()[0], w.shape()[1].checked_mul(group))
    } else {
        (
            w.shape()[1]
                .checked_mul(group)
                .ok_or_else(|| error("convolution channels overflow"))?,
            Some(w.shape()[0]),
        )
    };
    let output_channels = output_channels.ok_or_else(|| error("convolution channels overflow"))?;
    if x.shape()[1] != input_channels || input_channels % group != 0 || output_channels % group != 0
    {
        return Err(error("convolution grouped channels mismatch"));
    }
    let x_shape = shape(x);
    let w_shape = shape(w);
    // Check unsigned geometry before invoking RUDA's shared shape calculation.
    for i in 0..N {
        let kernel = (w_shape[i + 2] - 1)
            .checked_mul(options.dilation[i])
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| error("convolution kernel overflow"))?;
        let pad = options.padding[i]
            .checked_mul(2)
            .ok_or_else(|| error("convolution padding overflow"))?;
        let extent = if options.transposed {
            if options.output_padding[i] >= options.stride[i]
                && options.output_padding[i] >= options.dilation[i]
            {
                return Err(error("invalid transposed convolution output padding"));
            }
            x_shape[i + 2]
                .checked_sub(1)
                .and_then(|v| v.checked_mul(options.stride[i]))
                .and_then(|v| v.checked_add(kernel))
                .and_then(|v| v.checked_add(options.output_padding[i]))
                .and_then(|v| v.checked_sub(pad))
        } else {
            if options.output_padding[i] != 0 {
                return Err(error("ordinary convolution output padding must be zero"));
            }
            x_shape[i + 2]
                .checked_add(pad)
                .and_then(|v| v.checked_sub(kernel))
        };
        if extent.is_none() {
            return Err(error("invalid or overflowing convolution geometry"));
        }
    }
    let output = if options.transposed {
        calculate_conv_transpose_output_shape(
            &x_shape,
            &w_shape,
            &options.stride,
            &options.padding,
            &options.output_padding,
            &options.dilation,
            options.groups,
        )
    } else {
        calculate_conv_output_shape(
            &x_shape,
            &w_shape,
            &options.stride,
            &options.padding,
            &options.dilation,
        )
    }
    .map_err(|e| error(format!("{e:?}")))?;
    layout(output, x.dtype())
}

fn pool_layout(x: &TensorLayout, options: &Pool2dOptions) -> Result<TensorLayout> {
    floating(&[x.clone()])?;
    if x.shape().len() != 4
        || options.kernel.contains(&0)
        || options.stride.contains(&0)
        || options.dilation.contains(&0)
    {
        return Err(error("invalid pooling geometry"));
    }
    let x_shape = shape(x);
    for i in 0..2 {
        if options.padding[i] > options.kernel[i] / 2 {
            return Err(error("ACLNN pool padding exceeds half the kernel"));
        }
        let span = (options.kernel[i] - 1)
            .checked_mul(options.dilation[i])
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| error("pool kernel overflow"))?;
        let numerator = options.padding[i]
            .checked_mul(2)
            .and_then(|v| v.checked_add(x_shape[i + 2]))
            .and_then(|v| v.checked_sub(span))
            .ok_or_else(|| error("invalid pool extent"))?;
        if options.ceil_mode && numerator.checked_add(options.stride[i] - 1).is_none() {
            return Err(error("pool ceil geometry overflow"));
        }
    }
    let output = calculate_pool_output_shape(
        &x_shape,
        &options.kernel,
        &options.stride,
        &options.padding,
        &options.dilation,
        options.ceil_mode,
    )
    .map_err(|e| error(format!("{e:?}")))?;
    for i in 0..2 {
        if options.ceil_mode
            && (output[i + 2] - 1)
                .checked_mul(options.stride[i])
                .is_none_or(|v| v >= x_shape[i + 2] + options.padding[i])
        {
            return Err(error(
                "ACLNN ceil pool would omit a right-padding window required by RUDA",
            ));
        }
    }
    layout(output, x.dtype())
}

#[derive(Clone)]
enum Operation {
    Conv {
        stride: Vec<i64>,
        padding: Vec<i64>,
        dilation: Vec<i64>,
        output_padding: Vec<i64>,
        groups: i64,
        transposed: bool,
        backward: Option<usize>,
    },
    Pool {
        options: Pool2dOptions,
        max: bool,
        backward: bool,
    },
}
fn submit(
    client: &ComputeClient<AscendRuntime>,
    inputs: &[TensorBuffer],
    targets: Vec<TensorLayout>,
    operation: Operation,
) -> Result<Vec<TensorBuffer>> {
    let outputs: Vec<_> = targets.iter().map(|l| typed::allocate(client, l)).collect();
    if targets.iter().all(|l| l.byte_len() == 0) {
        return Ok(outputs);
    }
    let mut layouts = inputs
        .iter()
        .map(typed::layout)
        .collect::<Result<Vec<_>>>()?;
    layouts.extend(targets);
    client.flush().map_err(error)?;
    let guards = inputs
        .iter()
        .chain(&outputs)
        .map(|t| client.get_resource(t.handle.clone()).map_err(error))
        .collect::<Result<Vec<_>>>()?;
    let resources = guards
        .iter()
        .zip(&layouts)
        .map(|(g, l)| {
            let mut r = g.resource().clone();
            if r.byte_len() < l.byte_len() {
                return Err(error("spatial resource is too short"));
            }
            r.size = l.byte_len();
            Ok(r)
        })
        .collect::<Result<Vec<_>>>()?;
    let input_count = inputs.len();
    let result = WORKER
        .get()
        .ok_or_else(|| error("Ascend runtime is not initialized"))?
        .1
        .call(move |state| state.spatial(operation, input_count, layouts, resources));
    drop(guards);
    result?;
    Ok(outputs)
}

impl AscendRuntime {
    pub fn convolution<const N: usize>(
        client: &ComputeClient<Self>,
        x: TensorBuffer,
        weight: TensorBuffer,
        bias: Option<TensorBuffer>,
        options: SpatialConvOptions<N>,
    ) -> Result<TensorBuffer> {
        let target = conv_layout(&typed::layout(&x)?, &typed::layout(&weight)?, &options)?;
        let mut inputs = vec![x, weight];
        if let Some(bias) = bias {
            let bias_layout = typed::layout(&bias)?;
            if bias_layout.dtype() != target.dtype()
                || bias_layout.shape() != [target.shape()[1]]
                || !is_contiguous(&bias_layout)
            {
                return Err(error("convolution bias shape/dtype mismatch"));
            }
            inputs.push(bias);
        }
        Ok(submit(
            client,
            &inputs,
            vec![target],
            conv_operation(options, None)?,
        )?
        .remove(0))
    }
    pub fn convolution_backward<const N: usize>(
        client: &ComputeClient<Self>,
        x: TensorBuffer,
        weight: TensorBuffer,
        grad: TensorBuffer,
        options: SpatialConvOptions<N>,
        weight_gradient: bool,
    ) -> Result<TensorBuffer> {
        let xl = typed::layout(&x)?;
        let wl = typed::layout(&weight)?;
        let expected = conv_layout(&xl, &wl, &options)?;
        let gl = typed::layout(&grad)?;
        floating(&[xl.clone(), wl.clone(), gl.clone()])?;
        if expected.shape() != gl.shape() {
            return Err(error("convolution gradient shape mismatch"));
        }
        let targets = vec![
            TensorLayout::contiguous(xl.shape(), xl.dtype())?,
            TensorLayout::contiguous(wl.shape(), wl.dtype())?,
        ];
        let index = usize::from(weight_gradient);
        Ok(submit(
            client,
            &[grad, x, weight],
            targets,
            conv_operation(options, Some(index))?,
        )?
        .remove(index))
    }
    pub fn avg_pool2d(
        client: &ComputeClient<Self>,
        x: TensorBuffer,
        options: Pool2dOptions,
    ) -> Result<TensorBuffer> {
        if options.dilation != [1; 2] {
            return Err(error("average pooling has no dilation"));
        }
        let target = pool_layout(&typed::layout(&x)?, &options)?;
        Ok(submit(
            client,
            &[x],
            vec![target],
            Operation::Pool {
                options,
                max: false,
                backward: false,
            },
        )?
        .remove(0))
    }
    pub fn avg_pool2d_backward(
        client: &ComputeClient<Self>,
        x: TensorBuffer,
        grad: TensorBuffer,
        options: Pool2dOptions,
    ) -> Result<TensorBuffer> {
        let xl = typed::layout(&x)?;
        let gl = typed::layout(&grad)?;
        floating(&[xl.clone(), gl.clone()])?;
        if options.dilation != [1; 2] || pool_layout(&xl, &options)?.shape() != gl.shape() {
            return Err(error("average pool gradient geometry mismatch"));
        }
        let target = TensorLayout::contiguous(xl.shape(), xl.dtype())?;
        Ok(submit(
            client,
            &[grad, x],
            vec![target],
            Operation::Pool {
                options,
                max: false,
                backward: true,
            },
        )?
        .remove(0))
    }
    pub fn max_pool2d_with_indices(
        client: &ComputeClient<Self>,
        x: TensorBuffer,
        options: Pool2dOptions,
    ) -> Result<(TensorBuffer, TensorBuffer)> {
        let xl = typed::layout(&x)?;
        let target = pool_layout(&xl, &options)?;
        if xl.shape()[2]
            .checked_mul(xl.shape()[3])
            .is_none_or(|v| v > i32::MAX as i64 + 1)
        {
            return Err(error("pool plane does not fit I32 indices"));
        }
        let indices = TensorLayout::contiguous(target.shape(), DType::I32)?;
        let mut out = submit(
            client,
            &[x],
            vec![target, indices],
            Operation::Pool {
                options,
                max: true,
                backward: false,
            },
        )?;
        let indices = out.pop().unwrap();
        Ok((out.pop().unwrap(), indices))
    }
    pub fn max_pool2d_backward(
        client: &ComputeClient<Self>,
        x: TensorBuffer,
        grad: TensorBuffer,
        indices: TensorBuffer,
        options: Pool2dOptions,
    ) -> Result<TensorBuffer> {
        let xl = typed::layout(&x)?;
        let gl = typed::layout(&grad)?;
        let il = typed::layout(&indices)?;
        floating(&[xl.clone(), gl.clone()])?;
        if pool_layout(&xl, &options)?.shape() != gl.shape()
            || il.shape() != gl.shape()
            || !matches!(il.dtype(), DType::I32 | DType::I64)
            || !is_contiguous(&il)
        {
            return Err(error("max pool gradient/indices contract mismatch"));
        }
        let target = TensorLayout::contiguous(xl.shape(), xl.dtype())?;
        Ok(submit(
            client,
            &[grad, x, indices],
            vec![target],
            Operation::Pool {
                options,
                max: true,
                backward: true,
            },
        )?
        .remove(0))
    }
}
fn conv_operation<const N: usize>(
    o: SpatialConvOptions<N>,
    backward: Option<usize>,
) -> Result<Operation> {
    Ok(Operation::Conv {
        stride: ints(&o.stride)?,
        padding: ints(&o.padding)?,
        dilation: ints(&o.dilation)?,
        output_padding: ints(&o.output_padding)?,
        groups: i64::try_from(o.groups).map_err(error)?,
        transposed: o.transposed,
        backward,
    })
}

impl State {
    fn spatial(
        &mut self,
        operation: Operation,
        input_count: usize,
        layouts: Vec<TensorLayout>,
        resources: Vec<AscendResource>,
    ) -> Result<()> {
        if resources.len() != layouts.len()
            || resources
                .iter()
                .zip(&layouts)
                .any(|(r, l)| r.size != l.byte_len())
        {
            return Err(error("spatial binding contract mismatch"));
        }
        let addresses = resources
            .iter()
            .map(|r| self.pointer(r).map(|p| p as usize))
            .collect::<Result<Vec<_>>>()?;
        for (i, (&a, l)) in addresses.iter().zip(&layouts).enumerate() {
            let end = a
                .checked_add(l.byte_len())
                .ok_or_else(|| error("spatial address overflow"))?;
            if l.byte_len() != 0 && (a == 0 || a % l.dtype().bytes() != 0) {
                return Err(error("invalid spatial device address"));
            }
            if i >= input_count && l.byte_len() != 0 {
                for j in 0..i {
                    let other_end = addresses[j]
                        .checked_add(layouts[j].byte_len())
                        .ok_or_else(|| error("spatial address overflow"))?;
                    if layouts[j].byte_len() != 0 && a < other_end && addresses[j] < end {
                        return Err(error("spatial output aliases another binding"));
                    }
                }
            }
        }
        self.session.bind()?;
        let descriptors = layouts
            .iter()
            .cloned()
            .zip(addresses)
            .map(|(l, a)| unsafe {
                if l.shape().len() == 1 {
                    Descriptor::new(&self.session, l, a as *mut _)
                } else {
                    Descriptor::channels_first(&self.session, l, a as *mut _)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let tensor = |i: usize| descriptors[i].handle.as_ptr();
        unsafe {
            match operation {
                Operation::Conv {
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    groups,
                    transposed,
                    backward,
                } => {
                    let stride = self.session.int_array(&stride)?;
                    let padding = self.session.int_array(&padding)?;
                    let dilation = self.session.int_array(&dilation)?;
                    let output_padding = self.session.int_array(&output_padding)?;
                    if let Some(index) = backward {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclTensor,
                            *const AclTensor,
                            *const AclIntArray,
                            *const AclIntArray,
                            *const AclIntArray,
                            *const AclIntArray,
                            bool,
                            *const AclIntArray,
                            i32,
                            *const AclBoolArray,
                            i8,
                            *mut AclTensor,
                            *mut AclTensor,
                            *mut AclTensor,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let bias_sizes = self.session.int_array(&[if transposed {
                            layouts[2].shape()[1] * groups
                        } else {
                            layouts[2].shape()[0]
                        }])?;
                        let mask = self.session.bool_array(&[index == 0, index == 1, false])?;
                        let plan: Plan = self
                            .session
                            .ops
                            .get(c"aclnnConvolutionBackwardGetWorkspaceSize")?;
                        let run = self.session.ops.get(c"aclnnConvolutionBackward")?;
                        self.session
                            .execute("aclnnConvolutionBackward", run, |size, exec| {
                                plan(
                                    tensor(0),
                                    tensor(1),
                                    tensor(2),
                                    bias_sizes.handle.as_ptr(),
                                    stride.handle.as_ptr(),
                                    padding.handle.as_ptr(),
                                    dilation.handle.as_ptr(),
                                    transposed,
                                    output_padding.handle.as_ptr(),
                                    groups as i32,
                                    mask.handle.as_ptr(),
                                    0,
                                    tensor(3),
                                    tensor(4),
                                    std::ptr::null_mut(),
                                    size,
                                    exec,
                                )
                            })
                    } else {
                        type Plan = unsafe extern "C" fn(
                            *const AclTensor,
                            *const AclTensor,
                            *const AclTensor,
                            *const AclIntArray,
                            *const AclIntArray,
                            *const AclIntArray,
                            bool,
                            *const AclIntArray,
                            i64,
                            *mut AclTensor,
                            i8,
                            *mut u64,
                            *mut *mut AclOpExecutor,
                        ) -> i32;
                        let plan: Plan =
                            self.session.ops.get(c"aclnnConvolutionGetWorkspaceSize")?;
                        let run = self.session.ops.get(c"aclnnConvolution")?;
                        self.session.execute("aclnnConvolution", run, |size, exec| {
                            plan(
                                tensor(0),
                                tensor(1),
                                if input_count == 3 {
                                    tensor(2)
                                } else {
                                    std::ptr::null_mut()
                                },
                                stride.handle.as_ptr(),
                                padding.handle.as_ptr(),
                                dilation.handle.as_ptr(),
                                transposed,
                                output_padding.handle.as_ptr(),
                                groups,
                                tensor(input_count),
                                0,
                                size,
                                exec,
                            )
                        })
                    }
                }
                Operation::Pool {
                    options,
                    max,
                    backward,
                } => {
                    let kernel = self.session.int_array(&ints(&options.kernel)?)?;
                    let stride = self.session.int_array(&ints(&options.stride)?)?;
                    let padding = self.session.int_array(&ints(&options.padding)?)?;
                    let dilation = self.session.int_array(&ints(&options.dilation)?)?;
                    match (max, backward) {
                        (false, false) => {
                            type Plan = unsafe extern "C" fn(
                                *const AclTensor,
                                *const AclIntArray,
                                *const AclIntArray,
                                *const AclIntArray,
                                bool,
                                bool,
                                i64,
                                i8,
                                *mut AclTensor,
                                *mut u64,
                                *mut *mut AclOpExecutor,
                            ) -> i32;
                            let plan: Plan =
                                self.session.ops.get(c"aclnnAvgPool2dGetWorkspaceSize")?;
                            let run = self.session.ops.get(c"aclnnAvgPool2d")?;
                            self.session.execute("aclnnAvgPool2d", run, |size, exec| {
                                plan(
                                    tensor(0),
                                    kernel.handle.as_ptr(),
                                    stride.handle.as_ptr(),
                                    padding.handle.as_ptr(),
                                    options.ceil_mode,
                                    options.count_include_pad,
                                    0,
                                    0,
                                    tensor(1),
                                    size,
                                    exec,
                                )
                            })
                        }
                        (false, true) => {
                            type Plan = unsafe extern "C" fn(
                                *const AclTensor,
                                *const AclTensor,
                                *const AclIntArray,
                                *const AclIntArray,
                                *const AclIntArray,
                                bool,
                                bool,
                                i64,
                                i8,
                                *mut AclTensor,
                                *mut u64,
                                *mut *mut AclOpExecutor,
                            ) -> i32;
                            let plan: Plan = self
                                .session
                                .ops
                                .get(c"aclnnAvgPool2dBackwardGetWorkspaceSize")?;
                            let run = self.session.ops.get(c"aclnnAvgPool2dBackward")?;
                            self.session
                                .execute("aclnnAvgPool2dBackward", run, |size, exec| {
                                    plan(
                                        tensor(0),
                                        tensor(1),
                                        kernel.handle.as_ptr(),
                                        stride.handle.as_ptr(),
                                        padding.handle.as_ptr(),
                                        options.ceil_mode,
                                        options.count_include_pad,
                                        0,
                                        0,
                                        tensor(2),
                                        size,
                                        exec,
                                    )
                                })
                        }
                        (true, false) => {
                            type Plan = unsafe extern "C" fn(
                                *const AclTensor,
                                *const AclIntArray,
                                *const AclIntArray,
                                *const AclIntArray,
                                *const AclIntArray,
                                bool,
                                *mut AclTensor,
                                *mut AclTensor,
                                *mut u64,
                                *mut *mut AclOpExecutor,
                            ) -> i32;
                            let plan: Plan = self
                                .session
                                .ops
                                .get(c"aclnnMaxPool2dWithIndicesGetWorkspaceSize")?;
                            let run = self.session.ops.get(c"aclnnMaxPool2dWithIndices")?;
                            self.session
                                .execute("aclnnMaxPool2dWithIndices", run, |size, exec| {
                                    plan(
                                        tensor(0),
                                        kernel.handle.as_ptr(),
                                        stride.handle.as_ptr(),
                                        padding.handle.as_ptr(),
                                        dilation.handle.as_ptr(),
                                        options.ceil_mode,
                                        tensor(1),
                                        tensor(2),
                                        size,
                                        exec,
                                    )
                                })
                        }
                        (true, true) => {
                            type Plan = unsafe extern "C" fn(
                                *const AclTensor,
                                *const AclTensor,
                                *const AclTensor,
                                *const AclIntArray,
                                *const AclIntArray,
                                *const AclIntArray,
                                *const AclIntArray,
                                bool,
                                *mut AclTensor,
                                *mut u64,
                                *mut *mut AclOpExecutor,
                            ) -> i32;
                            let plan: Plan = self
                                .session
                                .ops
                                .get(c"aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize")?;
                            let run = self.session.ops.get(c"aclnnMaxPool2dWithIndicesBackward")?;
                            self.session.execute(
                                "aclnnMaxPool2dWithIndicesBackward",
                                run,
                                |size, exec| {
                                    plan(
                                        tensor(0),
                                        tensor(1),
                                        tensor(2),
                                        kernel.handle.as_ptr(),
                                        stride.handle.as_ptr(),
                                        padding.handle.as_ptr(),
                                        dilation.handle.as_ptr(),
                                        options.ceil_mode,
                                        tensor(3),
                                        size,
                                        exec,
                                    )
                                },
                            )
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_convolution_geometry_and_dtype() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let x = TensorLayout::contiguous(&[2, 4, 7, 9], dtype).unwrap();
            let w = TensorLayout::contiguous(&[6, 2, 3, 3], dtype).unwrap();
            let mut o = SpatialConvOptions {
                stride: [2, 2],
                padding: [1, 1],
                dilation: [1, 1],
                groups: 2,
                transposed: false,
                output_padding: [0, 0],
            };
            let out = conv_layout(&x, &w, &o).unwrap();
            assert_eq!(out.shape(), &[2, 6, 4, 5]);
            assert_eq!(out.dtype(), dtype);
            o.groups = 1;
            assert!(conv_layout(&x, &w, &o).is_err());
        }
    }
    #[test]
    fn volume_transpose_and_pool_ceil_contract() {
        let x = TensorLayout::contiguous(&[2, 4, 3, 5, 7], DType::F32).unwrap();
        let w = TensorLayout::contiguous(&[4, 3, 3, 3, 3], DType::F32).unwrap();
        let o = SpatialConvOptions {
            stride: [2; 3],
            padding: [1; 3],
            dilation: [1; 3],
            groups: 2,
            transposed: true,
            output_padding: [1; 3],
        };
        assert_eq!(conv_layout(&x, &w, &o).unwrap().shape(), &[2, 6, 6, 10, 14]);
        let x = TensorLayout::contiguous(&[1, 2, 4, 4], DType::F32).unwrap();
        let o = Pool2dOptions {
            kernel: [3; 2],
            stride: [3; 2],
            padding: [1; 2],
            dilation: [1; 2],
            ceil_mode: true,
            count_include_pad: false,
        };
        assert_eq!(pool_layout(&x, &o).unwrap().shape(), &[1, 2, 2, 2]);
        let x = TensorLayout::contiguous(&[1, 2, 3, 3], DType::F32).unwrap();
        let o = Pool2dOptions {
            kernel: [2; 2],
            stride: [2; 2],
            padding: [1; 2],
            ..o
        };
        assert!(pool_layout(&x, &o).is_err());
    }
}
