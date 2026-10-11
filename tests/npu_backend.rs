use rust_ascend::{
    Autodiff, RudaAscend,
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        Backend, DType, FloatDType, TensorData,
        api::{Bool, Int, Tensor, module, quantization::QuantizationParameters},
        ops::{
            BoolTensorOps, DeformConvOptions, FloatTensorOps, IntTensorOps, InterpolateMode,
            InterpolateOptions, ModuleOps,
        },
        quantization::{QuantParam, QuantScheme, QuantStore, QuantValue},
    },
};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a == b
                || a.is_nan() && b.is_nan()
                || a.is_finite() && b.is_finite() && (a - b).abs() <= 3e-5 * (1.0 + b.abs()),
            "index {index}: {a} != {b}"
        );
    }
}

#[test]
#[ignore = "requires an exclusive Ascend 950DT process and CANN_TOOLKIT_ROOT"]
fn ascend_backend_ops_on_950dt() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("CANN_TOOLKIT_ROOT").ok_or("CANN_TOOLKIT_ROOT is required")?;
    let options = RuntimeOptions::new(toolkit);
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    type AD = Autodiff<RudaAscend>;
    let x = Tensor::<AD, 4>::from_data(
        TensorData::new((1..=9).map(|v| v as f32).collect::<Vec<_>>(), [1, 1, 3, 3]),
        (&device, DType::F32),
    )
    .require_grad();
    let w = Tensor::<AD, 4>::ones([1, 1, 2, 2], (&device, DType::F32)).require_grad();
    let conv = module::conv2d(
        x.clone(),
        w.clone(),
        None,
        rust_ascend::tensor::ops::ConvOptions::new([1; 2], [0; 2], [1; 2], 1),
    );
    close(
        conv.clone().into_data().as_slice::<f32>()?,
        &[12., 16., 24., 28.],
    );
    let grads = conv.sum().backward();
    close(
        x.grad(&grads)
            .ok_or("missing convolution input gradient")?
            .into_data()
            .as_slice::<f32>()?,
        &[1., 2., 1., 2., 4., 2., 1., 2., 1.],
    );
    close(
        w.grad(&grads)
            .ok_or("missing convolution weight gradient")?
            .into_data()
            .as_slice::<f32>()?,
        &[12., 16., 24., 28.],
    );

    let offset = Tensor::<AD, 4>::zeros([1, 2, 3, 3], (&device, DType::F32)).require_grad();
    let weight = Tensor::<AD, 4>::full([1, 1, 1, 1], 2.0, (&device, DType::F32)).require_grad();
    let mask = Tensor::<AD, 4>::ones([1, 1, 3, 3], (&device, DType::F32)).require_grad();
    let bias = Tensor::<AD, 1>::ones([1], (&device, DType::F32)).require_grad();
    let deformed = module::deform_conv2d(
        x.clone(),
        offset.clone(),
        weight.clone(),
        Some(mask.clone()),
        Some(bias.clone()),
        DeformConvOptions::new([1; 2], [0; 2], [1; 2], 1, 1),
    );
    close(
        deformed.clone().into_data().as_slice::<f32>()?,
        &[3., 5., 7., 9., 11., 13., 15., 17., 19.],
    );
    let gradients = deformed.sum().backward();
    close(
        x.grad(&gradients)
            .ok_or("missing deform input gradient")?
            .into_data()
            .as_slice::<f32>()?,
        &[2.; 9],
    );
    close(
        weight
            .grad(&gradients)
            .ok_or("missing deform weight gradient")?
            .into_data()
            .as_slice::<f32>()?,
        &[45.],
    );
    close(
        bias.grad(&gradients)
            .ok_or("missing deform bias gradient")?
            .into_data()
            .as_slice::<f32>()?,
        &[9.],
    );
    close(
        mask.grad(&gradients)
            .ok_or("missing deform mask gradient")?
            .into_data()
            .as_slice::<f32>()?,
        &[2., 4., 6., 8., 10., 12., 14., 16., 18.],
    );
    close(
        offset
            .grad(&gradients)
            .ok_or("missing deform offset gradient")?
            .into_data()
            .as_slice::<f32>()?,
        &[
            6., 6., 6., 6., 6., 6., -14., -16., -18., 2., 2., -6., 2., 2., -12., 2., 2., -18.,
        ],
    );
    let fractional = module::deform_conv2d(
        Tensor::<RudaAscend, 4>::from_data(
            TensorData::new(vec![1., 2., 3., 4.], [1, 1, 2, 2]),
            (&device, DType::F32),
        ),
        Tensor::<RudaAscend, 4>::full([1, 2, 2, 2], 0.5, (&device, DType::F32)),
        Tensor::<RudaAscend, 4>::ones([1, 1, 1, 1], (&device, DType::F32)),
        None,
        None,
        DeformConvOptions::new([1; 2], [0; 2], [1; 2], 1, 1),
    );
    close(
        fractional.into_data().as_slice::<f32>()?,
        &[2.5, 1.5, 1.75, 1.0],
    );

    let integer = |values: Vec<i64>, shape: Vec<usize>| {
        RudaAscend::int_from_data(TensorData::new(values, shape), &device)
    };
    let read_int = |value| Tensor::<RudaAscend, 2, Int>::from_primitive(value).into_data();
    let product = RudaAscend::int_matmul(
        integer(vec![(1i64 << 40) + 3, 1], vec![1, 2]),
        integer(vec![2, 1], vec![2, 1]),
    );
    assert_eq!(read_int(product).as_slice::<i64>()?, &[(1i64 << 41) + 7]);
    let mean = RudaAscend::int_mean_dim(
        integer(vec![-5, 2, (1i64 << 40) + 1, (1i64 << 40) + 3], vec![2, 2]),
        1,
    );
    assert_eq!(read_int(mean).as_slice::<i64>()?, &[-1, (1i64 << 40) + 2]);
    let value = integer(vec![2, 3, 0, 5], vec![2, 2]);
    assert_eq!(
        read_int(RudaAscend::int_cumprod(value.clone(), 1)).as_slice::<i64>()?,
        &[2, 6, 0, 0]
    );
    assert_eq!(
        read_int(RudaAscend::int_cumprod(value, 0)).as_slice::<i64>()?,
        &[2, 3, 0, 15]
    );
    let value = integer(
        vec![(1i64 << 40) + 3, (1i64 << 40) + 1, (1i64 << 40) + 5],
        vec![1, 3],
    );
    assert_eq!(
        read_int(RudaAscend::int_cummin(value.clone(), 1)).as_slice::<i64>()?,
        &[(1i64 << 40) + 3, (1i64 << 40) + 1, (1i64 << 40) + 1]
    );
    assert_eq!(
        read_int(RudaAscend::int_cummax(value, 1)).as_slice::<i64>()?,
        &[(1i64 << 40) + 3, (1i64 << 40) + 3, (1i64 << 40) + 5]
    );
    let boolean = |values: Vec<bool>, shape: Vec<usize>| {
        RudaAscend::bool_from_data(TensorData::new(values, shape), &device)
    };
    let read_bool = |value| Tensor::<RudaAscend, 2, Bool>::from_primitive(value).into_data();
    let value = boolean(vec![true, false, false], vec![1, 3]);
    let indices = integer(vec![1, 1, 2], vec![1, 3]);
    let source = boolean(vec![false, true, false], vec![1, 3]);
    assert_eq!(
        read_bool(RudaAscend::bool_scatter_or(
            1,
            value.clone(),
            indices,
            source.clone()
        ))
        .as_slice::<bool>()?,
        &[true, true, false]
    );
    assert_eq!(
        read_bool(RudaAscend::bool_select_or(
            value,
            1,
            integer(vec![1, 1, 2], vec![3]),
            source
        ))
        .as_slice::<bool>()?,
        &[true, true, false]
    );

    let input = Tensor::<RudaAscend, 4>::from_data(
        TensorData::new((1..=9).map(|v| v as f32).collect::<Vec<_>>(), [1, 1, 3, 3]),
        (&device, DType::F32),
    );
    let (maximum, indices) =
        module::max_pool2d_with_indices(input.clone(), [2; 2], [2; 2], [1; 2], [1; 2], true);
    close(
        maximum.into_data().as_slice::<f32>()?,
        &[
            1.,
            3.,
            f32::NEG_INFINITY,
            7.,
            9.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
    );
    assert_eq!(
        indices.into_data().as_slice::<i32>()?,
        &[0, 2, -1, 6, 8, -1, -1, -1, -1]
    );
    close(
        module::avg_pool2d(input.clone(), [2; 2], [2; 2], [1; 2], true, true)
            .into_data()
            .as_slice::<f32>()?,
        &[0.25, 1.25, 0., 2.75, 7., 0., 0., 0., 0.],
    );
    close(
        module::adaptive_avg_pool2d(input, [2; 2])
            .into_data()
            .as_slice::<f32>()?,
        &[3., 4., 6., 7.],
    );
    let nan = Tensor::<RudaAscend, 4>::from_data(
        TensorData::new(vec![f32::NAN, 4., 3., 4.], [1, 1, 2, 2]),
        (&device, DType::F32),
    );
    let (maximum, indices) =
        module::max_pool2d_with_indices(nan, [2; 2], [1; 2], [0; 2], [1; 2], false);
    assert!(maximum.into_data().as_slice::<f32>()?[0].is_nan());
    assert_eq!(indices.into_data().as_slice::<i32>()?, &[0]);

    let signal = RudaAscend::float_from_data(TensorData::new(vec![1., 2., 3., 4.], [4]), &device);
    let (real, imag) = RudaAscend::rfft(signal, 0, Some(3));
    close(
        Tensor::<RudaAscend, 1>::from_primitive(rust_ascend::tensor::TensorPrimitive::Float(
            real.clone(),
        ))
        .into_data()
        .as_slice::<f32>()?,
        &[6., -2., 2.],
    );
    close(
        Tensor::<RudaAscend, 1>::from_primitive(rust_ascend::tensor::TensorPrimitive::Float(
            imag.clone(),
        ))
        .into_data()
        .as_slice::<f32>()?,
        &[0., -2., 0.],
    );
    let restored = RudaAscend::irfft(real, imag, 0, Some(3));
    close(
        Tensor::<RudaAscend, 1>::from_primitive(rust_ascend::tensor::TensorPrimitive::Float(
            restored,
        ))
        .into_data()
        .as_slice::<f32>()?,
        &[1., 2., 3.],
    );

    let input = Tensor::<RudaAscend, 1>::from_data(
        TensorData::new(vec![-63.5, -1., 0.5, 1., 63.5], [5]),
        (&device, DType::F32),
    );
    let scales = Tensor::<RudaAscend, 1>::from_data([0.5], (&device, DType::F32));
    let quantized = input.quantize(&QuantScheme::default(), QuantizationParameters { scales });
    let bytes = quantized.clone().into_data().bytes;
    assert_eq!(&bytes[..8], &[0x81, 0xfe, 1, 2, 0x7f, 0, 0, 0]);
    close(
        quantized
            .dequantize_with_dtype(FloatDType::F32)
            .into_data()
            .as_slice::<f32>()?,
        &[-63.5, -1., 0.5, 1., 63.5],
    );
    for (format, store) in [
        (QuantValue::E4M3, QuantStore::Native),
        (QuantValue::E5M2, QuantStore::Native),
        (QuantValue::E4M3, QuantStore::PackedU32(0)),
        (QuantValue::E5M2, QuantStore::PackedU32(0)),
        (QuantValue::E2M1, QuantStore::PackedNative(0)),
        (QuantValue::E2M1, QuantStore::PackedU32(0)),
    ] {
        let values = vec![-3.5f32, -1., -0., 0.25, 1.5, 2., 3.5];
        let codes: Vec<u8> = values
            .iter()
            .map(|&v| match format {
                QuantValue::E4M3 => ruda_core::e4m3::from_f32(v).to_bits(),
                QuantValue::E5M2 => ruda_core::e5m2::from_f32(v).to_bits(),
                QuantValue::E2M1 => ruda_core::e2m1::from_f32(v).to_bits(),
                _ => unreachable!(),
            })
            .collect();
        let expected: Vec<f32> = codes
            .iter()
            .map(|&v| match format {
                QuantValue::E4M3 => ruda_core::e4m3::from_bits(v).to_f32(),
                QuantValue::E5M2 => ruda_core::e5m2::from_bits(v).to_f32(),
                QuantValue::E2M1 => ruda_core::e2m1::from_bits(v).to_f32(),
                _ => unreachable!(),
            })
            .collect();
        let scheme = QuantScheme::default().with_value(format).with_store(store);
        let input =
            Tensor::<RudaAscend, 1>::from_data(TensorData::new(values, [7]), (&device, DType::F32));
        let scales = Tensor::<RudaAscend, 1>::ones([1], (&device, DType::F32));
        let q = input.quantize(&scheme, QuantizationParameters { scales });
        let bits = scheme.size_bits_value();
        let factor = scheme.num_quants();
        let bytes_per_word = scheme.size_bits_stored() / 8;
        let mut payload = Vec::new();
        for chunk in codes.chunks(factor) {
            let word = chunk.iter().enumerate().fold(0u32, |word, (lane, &code)| {
                word | ((code as u32) << (lane * bits))
            });
            payload.extend_from_slice(&word.to_le_bytes()[..bytes_per_word]);
        }
        assert_eq!(
            &q.clone().into_data().bytes[..payload.len()],
            payload.as_slice(),
            "{format:?}/{store:?}"
        );
        close(
            q.clone()
                .dequantize_with_dtype(FloatDType::F32)
                .into_data()
                .as_slice::<f32>()?,
            &expected,
        );
        let mut flipped = expected.clone();
        flipped.reverse();
        close(
            q.flip([0])
                .dequantize_with_dtype(FloatDType::F32)
                .into_data()
                .as_slice::<f32>()?,
            &flipped,
        );
    }
    for (param, scale, code, values) in [
        (QuantParam::UE8M0, 0.75, 127, [1.0, 2.0, 3.0]),
        (QuantParam::UE4M3, 1.5, 0x3c, [1.5, 3.0, 4.5]),
    ] {
        let scheme = QuantScheme::default().with_param(param);
        let input = Tensor::<RudaAscend, 1>::from_data(values, (&device, DType::F32));
        let scales = Tensor::<RudaAscend, 1>::from_data([scale], (&device, DType::F32));
        let q = input.quantize(&scheme, QuantizationParameters { scales });
        assert_eq!(&q.clone().into_data().bytes[..5], &[1, 2, 3, 0, code]);
        close(
            q.dequantize_with_dtype(FloatDType::F32)
                .into_data()
                .as_slice::<f32>()?,
            &values,
        );
    }
    for mode in [
        InterpolateMode::Nearest,
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
        InterpolateMode::Lanczos3,
    ] {
        let input = Tensor::<AD, 4>::full([1, 1, 2, 2], 3.0, (&device, DType::F32)).require_grad();
        let output = module::interpolate(
            input.clone(),
            [4, 4],
            InterpolateOptions::new(mode).with_align_corners(false),
        );
        close(output.clone().into_data().as_slice::<f32>()?, &[3.; 16]);
        let gradients = output.sum().backward();
        close(
            input
                .grad(&gradients)
                .ok_or("missing interpolation gradient")?
                .into_data()
                .as_slice::<f32>()?,
            &[4.; 4],
        );
    }
    let input = Tensor::<RudaAscend, 4>::from_data(
        TensorData::new(vec![1.0, 2., 3., 4.], [1, 1, 2, 2]),
        (&device, DType::F32),
    );
    let output = module::interpolate(
        input,
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(false),
    );
    close(
        output.into_data().as_slice::<f32>()?,
        &[
            1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
        ],
    );
    RudaAscend::sync(&device)?;
    Ok(())
}
