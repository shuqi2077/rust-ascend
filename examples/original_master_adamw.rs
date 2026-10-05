use rust_ascend::{
    RudaAscend,
    optim::{AdamWStorageStep, adamw_master_tensor_step},
    runtime::{AscendRuntime, RuntimeOptions},
    tensor::{DType, api::Tensor},
};

fn close(
    actual: &Tensor<RudaAscend, 1>,
    expected: &[f32; 4],
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = actual.clone().into_data().to_vec::<f32>()?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options = RuntimeOptions::new(toolkit);
    if let Some(value) = std::env::var_os("RUDA_CANN_LIBRARY") {
        options.acl_library = value;
    }
    if let Some(value) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&value)
            .map(|p| p.into_os_string())
            .collect();
    }
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    for dtype in [DType::F16, DType::BF16] {
        let mut parameter =
            Tensor::<RudaAscend, 1>::from_data([1., -1., 0., 2.], &device).cast(dtype);
        let mut master = parameter.clone().cast(DType::F32);
        let mut first = Tensor::<RudaAscend, 1>::zeros([4], &device).cast(DType::F32);
        let mut second = Tensor::<RudaAscend, 1>::zeros([4], &device).cast(DType::F32);
        let raw_gradient = [0.5f32, -1., 2., -0.25];
        let gradient = Tensor::<RudaAscend, 1>::from_data(raw_gradient, &device).cast(dtype);
        let mut expected_parameter = [1f32, -1., 0., 2.];
        let mut expected_first = [0f32; 4];
        let mut expected_second = [0f32; 4];
        for update in 1..=2 {
            let beta1 = 0.9f32;
            let beta2 = 0.999f32;
            let step = AdamWStorageStep {
                learning_rate: 0.01,
                beta1,
                beta2,
                epsilon: 1e-8,
                weight_decay: 0.02,
                correction1: 1. - beta1.powi(update),
                correction2: 1. - beta2.powi(update),
                inverse_gradient_scale: 0.5,
                clip_multiplier: 0.75,
            };
            for i in 0..4 {
                let g = (raw_gradient[i] * step.inverse_gradient_scale) * step.clip_multiplier;
                let m = beta1 * expected_first[i] + (1. - beta1) * g;
                let v = beta2 * expected_second[i] + (1. - beta2) * g * g;
                expected_parameter[i] = expected_parameter[i]
                    * (1. - step.learning_rate * step.weight_decay)
                    - step.learning_rate * (m / step.correction1)
                        / ((v / step.correction2).sqrt() + step.epsilon);
                expected_first[i] = m;
                expected_second[i] = v;
            }
            adamw_master_tensor_step(
                &mut parameter,
                &mut master,
                &gradient,
                &mut first,
                &mut second,
                step,
            )?;
            assert_eq!(parameter.dtype(), dtype);
            assert_eq!(master.dtype(), DType::F32);
            close(&master, &expected_parameter)?;
            close(&first, &expected_first)?;
            close(&second, &expected_second)?;
            assert_eq!(
                parameter
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()?,
                master
                    .clone()
                    .cast(dtype)
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()?
            );
            assert_eq!(
                gradient
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()?,
                raw_gradient.to_vec()
            );
        }
    }
    println!(
        "original RUDA AdamW kernel preserved FP32 master/moments across two half-storage updates"
    );
    Ok(())
}
