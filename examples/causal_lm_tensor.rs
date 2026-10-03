//! Generic causal loss with independent full-vocabulary references and shared LoRA gradients.
use rust_ascend::{
    Ascend, Autodiff, nn,
    runtime::{AscendDevice, AscendRuntime, RuntimeOptions},
    tensor::{
        DType, TensorData,
        api::{Int, Tensor},
    },
};
type AD = Autodiff<Ascend>;

fn close(actual: &[f32], expected: &[f64], name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err(format!("{name}: length mismatch").into());
    }
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a as f64 - b).abs() > 2e-4 + 2e-4 * b.abs() {
            return Err(format!("{name}[{i}]: {a} != {b}").into());
        }
    }
    Ok(())
}

fn identity(
    device: &AscendDevice,
    chunk: usize,
    shift: bool,
    dtype: DType,
    all_ignored: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (batch, time, vocab) = (2, 5, 65);
    let x: Vec<f32> = (0..batch * time * vocab)
        .map(|i| (i % 23) as f32 / 8. - 7.)
        .collect();
    let labels: Vec<i64> = (0..batch * time)
        .map(|i| {
            if all_ignored || i % 3 == 0 {
                -100
            } else {
                (i * 7 % vocab) as i64
            }
        })
        .collect();
    let hidden = Tensor::<AD, 3>::from_data(
        TensorData::new(x.clone(), [batch, time, vocab]),
        (device, DType::F32),
    )
    .require_grad();
    let target = Tensor::<AD, 2, Int>::from_data(
        TensorData::new(labels.clone(), [batch, time]),
        (device, dtype),
    );
    let result = nn::CausalCrossEntropyConfig {
        token_chunk_size: chunk,
        ignore_index: -100,
        shift,
    }
    .forward_hidden(hidden.clone(), target, Ok)?;
    let observed_sum = result.loss_sum.clone().into_data();
    let observed_count = result.valid_tokens.clone().into_data();
    let mean = result.mean()?;
    let observed_mean = mean.clone().into_data();
    let gradients = (mean.clone() + mean).backward();
    let dx = hidden
        .grad(&gradients)
        .ok_or("missing causal hidden gradient")?
        .into_data();
    let mut sum = 0f64;
    let mut valid = 0usize;
    let mut expected = vec![0f64; x.len()];
    for b in 0..batch {
        for t in 0..if shift { time - 1 } else { time } {
            let label = labels[b * time + t + usize::from(shift)];
            if label == -100 {
                continue;
            }
            valid += 1;
            let offset = (b * time + t) * vocab;
            let values = &x[offset..offset + vocab];
            let max = values
                .iter()
                .map(|&v| v as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = values.iter().map(|&v| (v as f64 - max).exp()).collect();
            let denominator: f64 = exp.iter().sum();
            sum += denominator.ln() + max - values[label as usize] as f64;
            for c in 0..vocab {
                expected[offset + c] =
                    2. * (exp[c] / denominator - if c == label as usize { 1. } else { 0. });
            }
        }
    }
    for value in &mut expected {
        *value /= valid.max(1) as f64;
    }
    close(observed_sum.as_slice::<f32>()?, &[sum], "causal loss sum")?;
    close(
        observed_count.as_slice::<f32>()?,
        &[valid as f64],
        "causal valid count",
    )?;
    close(
        observed_mean.as_slice::<f32>()?,
        &[sum / valid.max(1) as f64],
        "causal mean",
    )?;
    close(
        dx.as_slice::<f32>()?,
        &expected,
        "causal shifted hidden derivative",
    )?;
    println!(
        "ASCEND_CAUSAL_IDENTITY chunk={chunk} shift={shift} labels={dtype:?} all_ignored={all_ignored} passed=true"
    );
    Ok(())
}

struct DecoderHead {
    hidden: Tensor<AD, 3>,
    base: Tensor<Ascend, 2>,
    down: Tensor<AD, 2>,
    up: Tensor<AD, 2>,
    mode: u8,
}
impl nn::CausalLanguageModel<AD> for DecoderHead {
    fn forward_hidden(
        &self,
        _: Tensor<AD, 2, Int>,
    ) -> Result<Tensor<AD, 3>, rust_ascend::driver::CannError> {
        Ok(self.hidden.clone())
    }
    fn project(
        &self,
        hidden: Tensor<AD, 2>,
    ) -> Result<Tensor<AD, 2>, rust_ascend::driver::CannError> {
        match self.mode {
            0 => nn::linear_padded_bf16_fp32(hidden, self.down.clone()),
            1 => nn::linear_frozen_padded_bf16_fp32(hidden, self.base.clone()),
            _ => nn::lora_frozen_padded_linear_bf16_fp32(
                hidden,
                self.base.clone(),
                self.down.clone(),
                self.up.clone(),
                0.25,
            ),
        }
    }
}
fn projection(
    device: &AscendDevice,
    chunk: usize,
    mode: u8,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let (batch, time, width, vocab, rank) = (2, 5, 7, 9, 3);
    let hidden = Tensor::<AD, 3>::from_data(
        TensorData::new(
            (0..batch * time * width)
                .map(|i| (i % 13) as f32 / 32. - 0.25)
                .collect::<Vec<_>>(),
            [batch, time, width],
        ),
        (device, DType::F32),
    )
    .require_grad();
    let make = |shape: [usize; 2]| {
        Tensor::<AD, 2>::from_data(
            TensorData::new(
                (0..shape[0] * shape[1])
                    .map(|i| (i % 11) as f32 / 16. - 0.25)
                    .collect::<Vec<_>>(),
                shape,
            ),
            (device, DType::F32),
        )
        .require_grad()
    };
    let base = Tensor::<Ascend, 2>::from_data(
        TensorData::new(
            (0..vocab * width)
                .map(|i| (i % 7) as f32 / 16. - 0.125)
                .collect::<Vec<_>>(),
            [vocab, width],
        ),
        (device, DType::BF16),
    );
    let model = DecoderHead {
        hidden,
        base,
        down: make([if mode == 0 { vocab } else { rank }, width]),
        up: make([vocab, rank]),
        mode,
    };
    let tokens = Tensor::<AD, 2, Int>::from_data(
        TensorData::new(vec![0i32; batch * time], [batch, time]),
        device,
    );
    let labels = Tensor::<AD, 2, Int>::from_data(
        TensorData::new(vec![0i32, 2, -100, 1, 4, 8, 0, 6, -100, 3], [batch, time]),
        device,
    );
    let loss = nn::CausalCrossEntropyConfig {
        token_chunk_size: chunk,
        ..Default::default()
    }
    .forward_model(&model, tokens, labels)?
    .mean()?;
    let output = loss.clone().into_data().as_slice::<f32>()?.to_vec();
    let gradients = loss.backward();
    let mut values = vec![
        output,
        model
            .hidden
            .grad(&gradients)
            .ok_or("missing projection hidden gradient")?
            .into_data()
            .as_slice::<f32>()?
            .to_vec(),
    ];
    if mode != 1 {
        values.push(
            model
                .down
                .grad(&gradients)
                .ok_or("missing dense/LoRA down gradient")?
                .into_data()
                .as_slice::<f32>()?
                .to_vec(),
        );
    }
    if mode == 2 {
        values.push(
            model
                .up
                .grad(&gradients)
                .ok_or("missing LoRA up gradient")?
                .into_data()
                .as_slice::<f32>()?
                .to_vec(),
        );
    }
    Ok(values)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolkit = std::env::var_os("ASCEND_HOME_PATH").ok_or("set ASCEND_HOME_PATH")?;
    let mut options = RuntimeOptions::new(toolkit);
    if let Some(path) = std::env::var_os("RUDA_CANN_LIBRARY") {
        options.acl_library = path;
    }
    if let Some(paths) = std::env::var_os("RUDA_CANN_OPERATOR_LIBRARIES") {
        options.operator_libraries = std::env::split_paths(&paths)
            .map(|p| p.into_os_string())
            .collect();
    }
    // SAFETY: this standalone executable is the sole ACL context owner.
    let device = unsafe { AscendRuntime::initialize_exclusive(options)? };
    for chunk in [1, 3, 64] {
        for shift in [false, true] {
            for dtype in [DType::I32, DType::I64] {
                for all_ignored in [false, true] {
                    identity(&device, chunk, shift, dtype, all_ignored)?;
                }
            }
        }
    }
    for mode in 0..3 {
        let reference = projection(&device, 64, mode)?;
        for chunk in [1, 3] {
            let actual = projection(&device, chunk, mode)?;
            for (actual, expected) in actual.iter().zip(&reference) {
                close(
                    actual,
                    &expected.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    "chunk/full projection gradient",
                )?;
            }
            println!("ASCEND_CAUSAL_PROJECTION chunk={chunk} mode={mode} passed=true");
        }
    }
    println!("ASCEND_CAUSAL_LM_DEVICE_OK cases=30");
    Ok(())
}
