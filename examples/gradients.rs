use rust_ascend::driver::{CannDevice, tensor::{CannSession, DType}};
use std::{error::Error, ffi::OsString};

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite() && (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "element {index}: device={actual}, expected={expected}");
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let acl = std::env::var_os("RUDA_CANN_LIBRARY").unwrap_or_else(|| "libascendcl.so".into());
    let libraries: Vec<OsString> = match std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        Some(value) => std::env::split_paths(&value).map(|p| p.into_os_string()).collect(),
        None => vec!["libnnopbase.so".into(), "libopapi_math.so".into(), "libopapi_nn.so".into()],
    };
    // SAFETY: standalone executable owns ACL; paths must point to the installed trusted SDK.
    let session = unsafe { CannSession::open_exclusive_libraries(CannDevice::new(0)?, acl, libraries)? };
    let x = [-2.0f32, -1.0, 0.5, 1.0, 2.0, 3.0];
    let gamma = [0.5f32, 1.0, 2.0];
    let dy = [1.0f32, 0.5, -0.25, -0.5, 1.0, 0.25];
    let host_x = session.from_f32(&[2, 3], &x)?;
    let host_gamma = session.from_f32(&[3], &gamma)?;
    let host_dy = session.from_f32(&[2, 3], &dy)?;
    let epsilon = 1e-5;
    let mut expected_dx = [0.0f32; 6];
    let mut expected_dgamma = [0.0f32; 3];
    for row in 0..2 {
        let start = row * 3;
        let rstd = (x[start..start + 3].iter().map(|v| v * v).sum::<f32>() / 3.0 + epsilon).sqrt().recip();
        let dot = (0..3).map(|j| x[start + j] * dy[start + j] * gamma[j]).sum::<f32>() / 3.0;
        for j in 0..3 {
            expected_dx[start + j] = (dy[start + j] * gamma[j] - x[start + j] * dot * rstd * rstd) * rstd;
            expected_dgamma[j] += dy[start + j] * x[start + j] * rstd;
        }
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let input = session.cast(&host_x, dtype)?;
        let weight = session.cast(&host_gamma, dtype)?;
        let grad = session.cast(&host_dy, dtype)?;
        let (_, rstd) = session.rms_norm(&input, &weight, epsilon as f64)?;
        let (dx, dgamma) = session.rms_norm_backward(&input, &weight, &grad, &rstd)?;
        let tolerance = match dtype { DType::F32 => 1e-4, DType::F16 => 2e-3, _ => 2e-2 };
        close(&session.cast(&dx, DType::F32)?.to_f32()?, &expected_dx, tolerance);
        close(&dgamma.to_f32()?, &expected_dgamma, tolerance);
        let silu_grad = session.silu_backward(&input, &grad)?;
        let expected_silu: Vec<_> = x.iter().zip(dy).map(|(&x, dy)| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            dy * sigmoid * (1.0 + x * (1.0 - sigmoid))
        }).collect();
        close(&session.cast(&silu_grad, DType::F32)?.to_f32()?, &expected_silu, tolerance);
        for logarithmic in [false, true] {
            let output = if logarithmic { session.log_softmax(&input, -1)? } else { session.softmax(&input, -1)? };
            let saved = session.cast(&output, DType::F32)?.to_f32()?;
            let backward = if logarithmic { session.log_softmax_backward(&output, &grad, -1)? }
                else { session.softmax_backward(&output, &grad, -1)? };
            let mut expected = [0.0f32; 6];
            for row in 0..2 {
                let start = row * 3;
                let dot = (0..3).map(|j| dy[start + j] * if logarithmic { 1.0 } else { saved[start + j] }).sum::<f32>();
                for j in 0..3 {
                    expected[start + j] = if logarithmic { dy[start + j] - saved[start + j].exp() * dot }
                        else { saved[start + j] * (dy[start + j] - dot) };
                }
            }
            close(&session.cast(&backward, DType::F32)?.to_f32()?, &expected, tolerance);
        }
        println!("ASCEND_GRADIENTS_DEVICE_OK dtype={dtype:?}");
    }
    Ok(())
}
