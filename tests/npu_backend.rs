use rust_ascend::{
    Autodiff, RudaAscend,
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{
        Backend, DType, FloatDType, TensorData,
        api::{Tensor, module, quantization::QuantizationParameters},
        ops::{FloatTensorOps, ModuleOps},
        quantization::QuantScheme,
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
    RudaAscend::sync(&device)?;
    Ok(())
}
