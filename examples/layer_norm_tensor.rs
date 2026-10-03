//! RUDA's existing LayerNorm module and autodiff, executed through Ascend kernels.
use rust_ascend::{Ascend, Autodiff, runtime::{AscendRuntime, RuntimeOptions},
    tensor::{DType, TensorData, api::Tensor}};
use ruda_nn::LayerNormConfig;

type AD = Autodiff<Ascend>;

fn close(actual: &[f32], expected: &[f64], name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() { return Err(format!("{name}: length mismatch").into()); }
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64-b).abs() > 5e-4+5e-4*b.abs() {
            return Err(format!("{name}[{i}]: {a} != {b}").into());
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
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    let cases=[(0,32),(1,32),(3,96),(8,256),(3,4096),(0,4128),(1,8192),(3,8224),
        (0,7),(1,1),(3,2),(3,7),(3,31),(3,33),(3,65),(3,4095),(3,4097),(3,8225)];
    for (rows, width) in cases {
        for bias in [false, true] {
            let eps = 1e-3;
            let mut layer = LayerNormConfig::new(width).with_epsilon(eps).with_bias(bias).init::<AD>(&device);
            let weight: Vec<f32> = (0..width).map(|i| 0.5+(i%7) as f32/9.).collect();
            layer.gamma = layer.gamma.map(|_| Tensor::<AD, 1>::from_data(
                TensorData::new(weight.clone(), [width]), (&device, DType::F32)).require_grad());
            let values: Vec<f32> = (0..rows*width).map(|i| (i%31) as f32/7.-1.).collect();
            let upstream: Vec<f32> = (0..rows*width).map(|i| (i%11) as f32/5.-0.7).collect();
            // Rank three exercises flattening all leading dimensions, without changing the Tensor API.
            let x = Tensor::<AD, 3>::from_data(TensorData::new(values.clone(), [1, rows, width]),
                (&device, DType::F32)).require_grad();
            let dy = Tensor::<AD, 3>::from_data(TensorData::new(upstream.clone(), [1, rows, width]), (&device, DType::F32));
            let y = layer.forward(x.clone());
            let observed = y.clone().into_data();
            // Shared graph use must accumulate both contributions before native backward.
            let gradients = (y.clone()*dy.clone()+y*dy).backward();
            let dx = x.grad(&gradients).ok_or("missing input gradient")?.into_data();
            let dw = layer.gamma.val().grad(&gradients).ok_or("missing weight gradient")?.into_data();
            let mut expected = vec![0.; rows*width]; let mut expected_dx = expected.clone();
            let mut expected_dw = vec![0.; width]; let mut expected_db = vec![0.; width];
            for r in 0..rows {
                let row = &values[r*width..(r+1)*width];
                let mean = row.iter().map(|&v| v as f64).sum::<f64>()/width as f64;
                let variance = row.iter().map(|&v| (v as f64-mean).powi(2)).sum::<f64>()/width as f64;
                let rstd = (variance+eps).sqrt().recip();
                let normalized: Vec<_> = row.iter().map(|&v| (v as f64-mean)*rstd).collect();
                let g: Vec<_> = (0..width).map(|c| 2.*upstream[r*width+c] as f64*weight[c] as f64).collect();
                let mean_g = g.iter().sum::<f64>()/width as f64;
                let mean_gn = g.iter().zip(&normalized).map(|(g,n)| g*n).sum::<f64>()/width as f64;
                for c in 0..width {
                    let i = r*width+c;
                    expected[i] = normalized[c]*weight[c] as f64;
                    expected_dx[i] = (g[c]-mean_g-normalized[c]*mean_gn)*rstd;
                    expected_dw[c] += 2.*upstream[i] as f64*normalized[c];
                    expected_db[c] += 2.*upstream[i] as f64;
                }
            }
            close(observed.as_slice::<f32>()?, &expected, "Y")?;
            close(dx.as_slice::<f32>()?, &expected_dx, "dX")?;
            close(dw.as_slice::<f32>()?, &expected_dw, "dWeight")?;
            if let Some(beta) = &layer.beta {
                let db = beta.val().grad(&gradients).ok_or("missing bias gradient")?.into_data();
                close(db.as_slice::<f32>()?, &expected_db, "dBias")?;
            }
            println!("ASCEND_LAYER_NORM_TENSOR_CASE rows={rows} width={width} bias={bias} passed=true");
        }
    }
    println!("ASCEND_LAYER_NORM_TENSOR_DEVICE_OK cases={}",cases.len()*2);
    Ok(())
}
