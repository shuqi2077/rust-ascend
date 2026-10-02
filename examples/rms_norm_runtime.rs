//! Common-IR RMSNorm forward and both gradients through RUDA's ComputeClient.
use rust_ascend::{core::tensor::{DType, Shape, Strides},
    runtime::{AscendRuntime, ComputeClient, RuntimeOptions, TensorBuffer, portable::backend::Runtime}};

fn upload(client: &ComputeClient<AscendRuntime>, shape: &[usize], values: &[f32]) -> TensorBuffer {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for (i, &dim) in shape.iter().enumerate().rev() { strides[i] = stride; stride *= dim; }
    TensorBuffer { handle: client.create_from_slice(&bytes), shape: Shape::from(shape.to_vec()),
        strides: Strides::from(strides), dtype: DType::F32 }
}

fn close(client: &ComputeClient<AscendRuntime>, tensor: TensorBuffer, expected: &[f64], name: &str)
    -> Result<(), Box<dyn std::error::Error>> {
    let bytes = client.read_one(tensor.handle)?;
    if bytes.len() != expected.len()*4 { return Err(format!("{name}: byte length mismatch").into()); }
    for (i, (chunk, &want)) in bytes.chunks_exact(4).zip(expected).enumerate() {
        let actual = f32::from_ne_bytes(chunk.try_into().unwrap()) as f64;
        if !actual.is_finite() || (actual-want).abs() > 5e-4+5e-4*want.abs() {
            return Err(format!("{name}[{i}]: {actual} != {want}").into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options = RuntimeOptions::new(toolkit);
    if let Some(path) = std::env::var_os("RUDA_CANN_LIBRARY") { options.acl_library = path; }
    if let Some(paths) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&paths).map(|p| p.into_os_string()).collect();
    }
    // SAFETY: this standalone executable is the sole owner of the ACL context.
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    let client = AscendRuntime::client(&device);
    for (rows, width) in [(0, 32), (1, 32), (3, 96), (7, 256), (33, 4096)] {
        let eps = 1e-3f32;
        let x: Vec<f32> = (0..rows*width).map(|i| (i%31) as f32/7.-1.).collect();
        let weight: Vec<f32> = (0..width).map(|i| 0.5+(i%7) as f32/9.).collect();
        let dy: Vec<f32> = (0..rows*width).map(|i| (i%11) as f32/5.-0.7).collect();
        let input = upload(&client, &[1, rows, width], &x);
        let w = upload(&client, &[width], &weight);
        let grad = upload(&client, &[1, rows, width], &dy);
        let [output, rstd] = AscendRuntime::rms_norm(&client, input.clone(), w.clone(), eps as f64)?;
        let [dx, dw] = AscendRuntime::rms_norm_backward(&client, input, w, grad, rstd.clone())?;
        if &output.shape[..] != [1, rows, width] || &dx.shape[..] != [1, rows, width]
            || &dw.shape[..] != [width] || &rstd.shape[..] != [rows] {
            return Err("RMSNorm output shape mismatch".into());
        }
        let mut expected = vec![0.; x.len()];
        let mut expected_dx = expected.clone();
        let mut expected_dw = vec![0.; width];
        let mut expected_rstd = vec![0.; rows];
        for row in 0..rows {
            let offset = row*width;
            let a = &x[offset..offset+width];
            let r = (a.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()/width as f64+eps as f64).sqrt().recip();
            expected_rstd[row] = r;
            let mean_gx = (0..width).map(|c| dy[offset+c] as f64*weight[c] as f64*a[c] as f64)
                .sum::<f64>()/width as f64;
            for c in 0..width {
                let i = offset+c;
                let normalized = a[c] as f64*r;
                expected[i] = normalized*weight[c] as f64;
                expected_dx[i] = r*(dy[i] as f64*weight[c] as f64-a[c] as f64*r*r*mean_gx);
                expected_dw[c] += dy[i] as f64*normalized;
            }
        }
        close(&client, output, &expected, "Y")?;
        close(&client, rstd, &expected_rstd, "rstd")?;
        close(&client, dx, &expected_dx, "dX")?;
        close(&client, dw, &expected_dw, "dWeight")?;
        println!("ASCEND_RMS_NORM_RUNTIME_CASE rows={rows} width={width} passed=true");
    }
    client.flush()?;
    println!("ASCEND_RMS_NORM_RUNTIME_DEVICE_OK cases=5");
    Ok(())
}
