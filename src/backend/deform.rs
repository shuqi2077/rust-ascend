use super::*;

struct Geometry {
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    outputs: usize,
    kernel: [usize; 2],
    output: [usize; 2],
    area: usize,
    spatial: usize,
    dtype: FloatDType,
    options: DeformConvOptions<2>,
}
impl Geometry {
    fn new(
        x: &Primitive,
        offset: &Primitive,
        weight: &Primitive,
        mask: Option<&Primitive>,
        bias: Option<&Primitive>,
        options: DeformConvOptions<2>,
    ) -> Self {
        spatial::same_queue(x, &[offset, weight]);
        assert!(
            native_float(x),
            "deformable convolution requires F32/F16/BF16"
        );
        let [batch, channels, height, width] = x.meta.shape().dims();
        let [outputs, per_group, kh, kw] = weight.meta.shape().dims();
        assert!(options.weight_groups > 0 && options.offset_groups > 0);
        assert_eq!(channels % options.weight_groups, 0);
        assert_eq!(outputs % options.weight_groups, 0);
        assert_eq!(channels % options.offset_groups, 0);
        assert_eq!(per_group, channels / options.weight_groups);
        let area = kh.checked_mul(kw).expect("deformable kernel size overflow");
        height
            .checked_mul(width)
            .expect("deformable input size overflow");
        let output = options.output_size([height, width], [kh, kw]);
        let spatial = output[0]
            .checked_mul(output[1])
            .expect("deformable output size overflow");
        let offset_channels = options
            .offset_groups
            .checked_mul(area)
            .and_then(|v| v.checked_mul(2))
            .expect("deformable offset channel overflow");
        assert_eq!(
            offset.meta.shape(),
            &Shape::new([batch, offset_channels, output[0], output[1]])
        );
        assert_eq!(offset.dtype, x.dtype);
        assert_eq!(weight.dtype, x.dtype);
        if let Some(mask) = mask {
            spatial::same_queue(x, &[mask]);
            assert_eq!(mask.dtype, x.dtype);
            assert_eq!(
                mask.meta.shape(),
                &Shape::new([batch, offset_channels / 2, output[0], output[1]])
            );
        }
        if let Some(bias) = bias {
            spatial::same_queue(x, &[bias]);
            assert_eq!(bias.dtype, x.dtype);
            assert_eq!(bias.meta.shape(), &Shape::new([outputs]));
        }
        channels
            .checked_mul(area)
            .expect("deformable column size overflow");
        batch
            .checked_mul(spatial)
            .expect("deformable column size overflow");
        Self {
            batch,
            channels,
            height,
            width,
            outputs,
            kernel: [kh, kw],
            output,
            area,
            spatial,
            dtype: x.dtype.into(),
            options,
        }
    }
    fn channel_indices(&self, k: usize, direction: Option<usize>) -> Vec<i64> {
        (0..self.channels)
            .map(|channel| {
                let group = channel / (self.channels / self.options.offset_groups);
                let position = group * self.area + k;
                i64::try_from(direction.map_or(position, |d| position * 2 + d))
                    .expect("offset index overflow")
            })
            .collect()
    }
    fn field(&self, value: &Primitive, k: usize, direction: Option<usize>) -> Primitive {
        let indices = self.channel_indices(k, direction);
        let indices =
            RudaAscend::int_from_data(TensorData::new(indices, [self.channels]), &value.device);
        RudaAscend::float_reshape(
            RudaAscend::float_select(value.clone(), 1, indices),
            Shape::new([self.batch, self.channels, self.spatial]),
        )
    }
    fn coordinates(&self, offset: &Primitive, k: usize) -> [Primitive; 2] {
        std::array::from_fn(|axis| {
            let kernel_position = if axis == 0 {
                k / self.kernel[1]
            } else {
                k % self.kernel[1]
            };
            let grid: Vec<f32> = (0..self.spatial)
                .map(|position| {
                    let index = if axis == 0 {
                        position / self.output[1]
                    } else {
                        position % self.output[1]
                    };
                    index
                        .checked_mul(self.options.stride[axis])
                        .and_then(|v| {
                            kernel_position
                                .checked_mul(self.options.dilation[axis])
                                .and_then(|p| v.checked_add(p))
                        })
                        .expect("deformable coordinate overflow") as f32
                })
                .collect();
            let base = RudaAscend::float_cast(
                RudaAscend::float_from_data(
                    TensorData::new(grid, [1, 1, self.spatial]),
                    &offset.device,
                ),
                self.dtype,
            );
            let base = RudaAscend::float_sub_scalar(
                base,
                Scalar::Float(self.options.padding[axis] as f32 as f64),
            );
            RudaAscend::float_add(base, self.field(offset, k, Some(axis)))
        })
    }
    fn mask(&self, reference: &Primitive, mask: Option<&Primitive>, k: usize) -> Primitive {
        mask.map_or_else(
            || scalar_like(reference, Scalar::Float(1.0)),
            |m| self.field(m, k, None),
        )
    }
    fn columns(&self, x: &Primitive, offset: &Primitive, mask: Option<&Primitive>) -> Primitive {
        let columns = (0..self.area)
            .map(|k| {
                let coords = self.coordinates(offset, k);
                let value = sample(self, x, coords).value;
                let value = RudaAscend::float_mul(self.mask(&value, mask, k), value);
                RudaAscend::float_reshape(
                    value,
                    Shape::new([self.batch, self.channels, 1, self.spatial]),
                )
            })
            .collect();
        let columns = RudaAscend::float_cat(columns, 2);
        let columns = RudaAscend::float_permute(columns, &[1, 2, 0, 3]);
        RudaAscend::float_reshape(
            columns,
            Shape::new([
                self.options.weight_groups,
                self.channels / self.options.weight_groups * self.area,
                self.batch * self.spatial,
            ]),
        )
    }
    fn channel_sum(&self, value: Primitive) -> Primitive {
        let per_group = self.channels / self.options.offset_groups;
        let value = RudaAscend::float_reshape(
            value,
            Shape::new([
                self.batch,
                self.options.offset_groups,
                per_group,
                self.spatial,
            ]),
        );
        let mut sum = RudaAscend::float_zeros(
            Shape::new([self.batch, self.options.offset_groups, 1, self.spatial]),
            &value.device,
            self.dtype,
        );
        for channel in 0..per_group {
            let part = RudaAscend::float_slice(
                value.clone(),
                &[
                    Slice::full(),
                    Slice::full(),
                    Slice::from(channel..channel + 1),
                    Slice::full(),
                ],
            );
            sum = RudaAscend::float_add(sum, part);
        }
        sum
    }
}
fn valid_position(value: &Primitive, length: usize) -> Primitive {
    RudaAscend::bool_and(
        RudaAscend::float_greater_equal_elem(value.clone(), Scalar::Float(0.0), BoolDType::U8),
        RudaAscend::float_lower_elem(
            value.clone(),
            Scalar::Float(length as f32 as f64),
            BoolDType::U8,
        ),
    )
}
fn safe_index(value: Primitive, valid: Primitive) -> Primitive {
    RudaAscend::float_into_int(
        RudaAscend::float_mask_fill(value, RudaAscend::bool_not(valid), Scalar::Float(0.0)),
        IntDType::I64,
    )
}
fn complement(value: Primitive) -> Primitive {
    RudaAscend::float_sub(scalar_like(&value, Scalar::Float(1.0)), value)
}
struct Sample {
    value: Primitive,
    dy: Primitive,
    dx: Primitive,
    coords: [Primitive; 2],
}
fn sample(g: &Geometry, x: &Primitive, coords: [Primitive; 2]) -> Sample {
    let yf = RudaAscend::float_cast(coords[0].clone(), FloatDType::F32);
    let xf = RudaAscend::float_cast(coords[1].clone(), FloatDType::F32);
    let y0 = RudaAscend::float_floor(yf.clone());
    let x0 = RudaAscend::float_floor(xf.clone());
    let ly = RudaAscend::float_sub(yf.clone(), y0.clone());
    let lx = RudaAscend::float_sub(xf.clone(), x0.clone());
    let y1 = RudaAscend::float_add_scalar(y0.clone(), Scalar::Float(1.0));
    let x1 = RudaAscend::float_add_scalar(x0.clone(), Scalar::Float(1.0));
    let ys = [y0, y1];
    let xs = [x0, x1];
    let flat = RudaAscend::float_reshape(
        x.clone(),
        Shape::new([g.batch, g.channels, g.height * g.width]),
    );
    let neighbors: [Primitive; 4] = std::array::from_fn(|n| {
        if g.height == 0 || g.width == 0 {
            return RudaAscend::float_zeros(coords[0].meta.shape().clone(), &x.device, g.dtype);
        }
        let vy = valid_position(&ys[n / 2], g.height);
        let vx = valid_position(&xs[n % 2], g.width);
        let index = RudaAscend::int_add(
            RudaAscend::int_mul_scalar(
                safe_index(ys[n / 2].clone(), vy.clone()),
                Scalar::Int(g.width as i64),
            ),
            safe_index(xs[n % 2].clone(), vx.clone()),
        );
        let value = RudaAscend::float_gather(2, flat.clone(), index);
        RudaAscend::float_mask_fill(
            value,
            RudaAscend::bool_not(RudaAscend::bool_and(vy, vx)),
            Scalar::Float(0.0),
        )
    });
    let yw = [complement(ly.clone()), ly.clone()];
    let xw = [complement(lx.clone()), lx.clone()];
    let terms: [Primitive; 4] = std::array::from_fn(|n| {
        let weight = RudaAscend::float_cast(
            RudaAscend::float_mul(yw[n / 2].clone(), xw[n % 2].clone()),
            g.dtype,
        );
        RudaAscend::float_mul(weight, neighbors[n].clone())
    });
    let value = RudaAscend::float_add(
        RudaAscend::float_add(
            RudaAscend::float_add(terms[0].clone(), terms[1].clone()),
            terms[2].clone(),
        ),
        terms[3].clone(),
    );
    let inside = |value: Primitive, length: usize| {
        RudaAscend::bool_and(
            RudaAscend::float_greater_elem(value.clone(), Scalar::Float(-1.0), BoolDType::U8),
            RudaAscend::float_lower_elem(value, Scalar::Float(length as f32 as f64), BoolDType::U8),
        )
    };
    let inside = RudaAscend::bool_and(inside(yf, g.height), inside(xf, g.width));
    let value =
        RudaAscend::float_mask_fill(value, RudaAscend::bool_not(inside), Scalar::Float(0.0));
    let ly = RudaAscend::float_cast(ly, g.dtype);
    let lx = RudaAscend::float_cast(lx, g.dtype);
    let dy = RudaAscend::float_add(
        RudaAscend::float_mul(
            lx.clone(),
            RudaAscend::float_sub(neighbors[3].clone(), neighbors[1].clone()),
        ),
        RudaAscend::float_mul(
            complement(lx),
            RudaAscend::float_sub(neighbors[2].clone(), neighbors[0].clone()),
        ),
    );
    let dx = RudaAscend::float_add(
        RudaAscend::float_mul(
            ly.clone(),
            RudaAscend::float_sub(neighbors[3].clone(), neighbors[2].clone()),
        ),
        RudaAscend::float_mul(
            complement(ly),
            RudaAscend::float_sub(neighbors[1].clone(), neighbors[0].clone()),
        ),
    );
    Sample {
        value,
        dy,
        dx,
        coords,
    }
}
fn input_gradient(
    g: &Geometry,
    mut result: Primitive,
    coords: [Primitive; 2],
    mask: Primitive,
    column: Primitive,
) -> Primitive {
    if g.height == 0 || g.width == 0 {
        return result;
    }
    let y0 = RudaAscend::float_floor(coords[0].clone());
    let x0 = RudaAscend::float_floor(coords[1].clone());
    for dy in -1..=1 {
        for dx in -1..=1 {
            let y = RudaAscend::float_add_scalar(y0.clone(), Scalar::Float(dy as f64));
            let x = RudaAscend::float_add_scalar(x0.clone(), Scalar::Float(dx as f64));
            let yd = RudaAscend::float_abs(RudaAscend::float_sub(coords[0].clone(), y.clone()));
            let xd = RudaAscend::float_abs(RudaAscend::float_sub(coords[1].clone(), x.clone()));
            let vy = valid_position(&y, g.height);
            let vx = valid_position(&x, g.width);
            let valid = RudaAscend::bool_and(
                RudaAscend::bool_and(vy.clone(), vx.clone()),
                RudaAscend::bool_and(
                    RudaAscend::float_lower_elem(yd.clone(), Scalar::Float(1.0), BoolDType::U8),
                    RudaAscend::float_lower_elem(xd.clone(), Scalar::Float(1.0), BoolDType::U8),
                ),
            );
            let index = RudaAscend::int_add(
                RudaAscend::int_mul_scalar(safe_index(y, vy), Scalar::Int(g.width as i64)),
                safe_index(x, vx),
            );
            let weight = RudaAscend::float_mul(complement(yd), complement(xd));
            let value =
                RudaAscend::float_mul(RudaAscend::float_mul(mask.clone(), weight), column.clone());
            let value =
                RudaAscend::float_mask_fill(value, RudaAscend::bool_not(valid), Scalar::Float(0.0));
            result = RudaAscend::float_scatter_add(
                2,
                result,
                index,
                RudaAscend::float_cast(value, FloatDType::F32),
            );
        }
    }
    result
}
pub(super) fn forward(
    x: Primitive,
    offset: Primitive,
    weight: Primitive,
    mask: Option<Primitive>,
    bias: Option<Primitive>,
    options: DeformConvOptions<2>,
) -> Primitive {
    let g = Geometry::new(&x, &offset, &weight, mask.as_ref(), bias.as_ref(), options);
    let columns = g.columns(&x, &offset, mask.as_ref());
    let weight = RudaAscend::float_reshape(
        weight,
        Shape::new([
            g.options.weight_groups,
            g.outputs / g.options.weight_groups,
            g.channels / g.options.weight_groups * g.area,
        ]),
    );
    let out = RudaAscend::float_matmul(weight, columns);
    let out = RudaAscend::float_reshape(
        out,
        Shape::new([g.outputs, g.batch, g.output[0], g.output[1]]),
    );
    let out = RudaAscend::float_swap_dims(out, 0, 1);
    match bias {
        Some(bias) => RudaAscend::float_add(
            out,
            RudaAscend::float_reshape(bias, Shape::new([1, g.outputs, 1, 1])),
        ),
        None => out,
    }
}
pub(super) fn backward(
    x: Primitive,
    offset: Primitive,
    weight: Primitive,
    mask: Option<Primitive>,
    bias: Option<Primitive>,
    grad: Primitive,
    options: DeformConvOptions<2>,
) -> DeformConv2dBackward<RudaAscend> {
    let g = Geometry::new(&x, &offset, &weight, mask.as_ref(), bias.as_ref(), options);
    spatial::same_queue(&x, &[&grad]);
    assert_eq!(grad.dtype, x.dtype);
    assert_eq!(
        grad.meta.shape(),
        &Shape::new([g.batch, g.outputs, g.output[0], g.output[1]])
    );
    let bias_grad = bias.map(|bias| {
        let value = RudaAscend::float_sum_dim(
            RudaAscend::float_sum_dim(RudaAscend::float_sum_dim(grad.clone(), 0), 2),
            3,
        );
        RudaAscend::float_reshape(value, bias.meta.shape().clone())
    });
    let grouped = RudaAscend::float_reshape(
        RudaAscend::float_swap_dims(grad, 0, 1),
        Shape::new([
            g.options.weight_groups,
            g.outputs / g.options.weight_groups,
            g.batch * g.spatial,
        ]),
    );
    let columns = g.columns(&x, &offset, mask.as_ref());
    let weight_grad =
        RudaAscend::float_matmul(grouped.clone(), RudaAscend::float_swap_dims(columns, 1, 2));
    let weight_grad = RudaAscend::float_reshape(weight_grad, weight.meta.shape().clone());
    let weight = RudaAscend::float_reshape(
        weight,
        Shape::new([
            g.options.weight_groups,
            g.outputs / g.options.weight_groups,
            g.channels / g.options.weight_groups * g.area,
        ]),
    );
    let columns = RudaAscend::float_matmul(RudaAscend::float_swap_dims(weight, 1, 2), grouped);
    let columns = RudaAscend::float_reshape(
        columns,
        Shape::new([g.channels, g.area, g.batch, g.spatial]),
    );
    let columns = RudaAscend::float_permute(columns, &[2, 0, 1, 3]);
    let mut x_grad = RudaAscend::float_zeros(
        Shape::new([g.batch, g.channels, g.height * g.width]),
        &x.device,
        FloatDType::F32,
    );
    let mut offsets = Vec::with_capacity(g.area);
    let mut masks = Vec::with_capacity(g.area);
    for k in 0..g.area {
        let column = RudaAscend::float_slice(
            columns.clone(),
            &[
                Slice::full(),
                Slice::full(),
                Slice::from(k..k + 1),
                Slice::full(),
            ],
        );
        let column =
            RudaAscend::float_reshape(column, Shape::new([g.batch, g.channels, g.spatial]));
        let sampled = sample(&g, &x, g.coordinates(&offset, k));
        let mask_value = g.mask(&column, mask.as_ref(), k);
        let dy = g.channel_sum(RudaAscend::float_mul(
            RudaAscend::float_mul(mask_value.clone(), sampled.dy),
            column.clone(),
        ));
        let dx = g.channel_sum(RudaAscend::float_mul(
            RudaAscend::float_mul(mask_value.clone(), sampled.dx),
            column.clone(),
        ));
        let offsets_k = RudaAscend::float_cat(vec![dy, dx], 2);
        offsets.push(RudaAscend::float_reshape(
            offsets_k,
            Shape::new([g.batch, g.options.offset_groups, 1, 2, g.spatial]),
        ));
        if mask.is_some() {
            masks.push(g.channel_sum(RudaAscend::float_mul(column.clone(), sampled.value)));
        }
        x_grad = input_gradient(&g, x_grad, sampled.coords, mask_value, column);
    }
    let offset_grad = RudaAscend::float_reshape(
        RudaAscend::float_cat(offsets, 2),
        offset.meta.shape().clone(),
    );
    let mask_grad = mask.map(|mask| {
        RudaAscend::float_reshape(RudaAscend::float_cat(masks, 2), mask.meta.shape().clone())
    });
    let x_grad = RudaAscend::float_cast(
        RudaAscend::float_reshape(x_grad, x.meta.shape().clone()),
        g.dtype,
    );
    DeformConv2dBackward {
        x_grad,
        offset_grad,
        weight_grad,
        mask_grad,
        bias_grad,
    }
}
