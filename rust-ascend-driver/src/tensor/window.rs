use super::{CannError, DType, TensorLayout, invalid};

/// Flatten a time window across batches, then select a contiguous token range.
/// Source is [B,T,H]; output is [count,H]. No batch boundary is treated as a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenWindow {
    pub time_start: usize,
    pub time_len: usize,
    pub start: usize,
    pub count: usize,
}

#[derive(Clone, Debug)]
pub(super) struct WindowPlan {
    pub source: TensorLayout,
    pub output: TensorLayout,
    pub window: TokenWindow,
    sequence: usize,
    row_bytes: usize,
}

impl WindowPlan {
    pub fn new(source: &TensorLayout, window: TokenWindow) -> Result<Self, CannError> {
        if source.shape().len() != 3
            || source.shape()[2] <= 0
            || !matches!(
                source.dtype(),
                DType::F32 | DType::F16 | DType::BF16 | DType::I32 | DType::I64
            )
        {
            return Err(invalid(
                "token window requires contiguous floating/integer [B,T,H] with positive H",
            ));
        }
        let [batch, sequence, width] = <[i64; 3]>::try_from(source.shape()).unwrap();
        let batch = usize::try_from(batch).map_err(|_| invalid("token batch overflow"))?;
        let sequence = usize::try_from(sequence).map_err(|_| invalid("token sequence overflow"))?;
        if window
            .time_start
            .checked_add(window.time_len)
            .is_none_or(|end| end > sequence)
            || batch.checked_mul(window.time_len).is_none_or(|tokens| {
                window
                    .start
                    .checked_add(window.count)
                    .is_none_or(|end| end > tokens)
            })
        {
            return Err(invalid("token window is outside its batch/time domain"));
        }
        let count = i64::try_from(window.count).map_err(|_| invalid("token count overflow"))?;
        let row_bytes = usize::try_from(width)
            .ok()
            .and_then(|n| n.checked_mul(source.dtype().bytes()))
            .ok_or_else(|| invalid("token row byte count overflow"))?;
        Ok(Self {
            source: source.clone(),
            output: TensorLayout::contiguous(&[count, width], source.dtype())?,
            window,
            sequence,
            row_bytes,
        })
    }

    /// Maximal contiguous source spans; all offsets are bytes and preserve integer bits.
    pub fn spans(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let mut copied = 0usize;
        std::iter::from_fn(move || {
            if copied == self.window.count {
                return None;
            }
            let token = self.window.start + copied;
            let batch = token / self.window.time_len;
            let time = token % self.window.time_len;
            let count = (self.window.time_len - time).min(self.window.count - copied);
            let source = (batch * self.sequence + self.window.time_start + time) * self.row_bytes;
            let target = copied * self.row_bytes;
            copied += count;
            Some((source, target, count * self.row_bytes))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_chunks_preserve_batches_integer_bits_and_scattered_gradients() {
        for dtype in [DType::F32, DType::BF16, DType::I32, DType::I64] {
            for (batch, time, width) in [(2, 5, 3), (3, 2, 1), (1, 7, 65)] {
                let source = TensorLayout::contiguous(&[batch, time, width], dtype).unwrap();
                for offset in [0, 1] {
                    let tokens = batch as usize * (time as usize - 1);
                    for chunk in [1, 2, 3, 8, 64] {
                        let mut restored = vec![0u8; source.byte_len()];
                        let original: Vec<u8> = (0..source.byte_len())
                            .map(|i| (i % 251 + 1) as u8)
                            .collect();
                        for start in (0..tokens).step_by(chunk) {
                            let window = TokenWindow {
                                time_start: offset,
                                time_len: time as usize - 1,
                                start,
                                count: chunk.min(tokens - start),
                            };
                            let plan = WindowPlan::new(&source, window).unwrap();
                            let mut output = vec![0; plan.output.byte_len()];
                            for (from, to, bytes) in plan.spans() {
                                output[to..to + bytes]
                                    .copy_from_slice(&original[from..from + bytes]);
                                restored[from..from + bytes]
                                    .copy_from_slice(&output[to..to + bytes]);
                            }
                            for row in 0..window.count {
                                let token = start + row;
                                let source_row = token / window.time_len * time as usize
                                    + offset
                                    + token % window.time_len;
                                let bytes = width as usize * dtype.bytes();
                                assert_eq!(
                                    &output[row * bytes..(row + 1) * bytes],
                                    &original[source_row * bytes..(source_row + 1) * bytes]
                                );
                            }
                        }
                        for b in 0..batch as usize {
                            for t in 0..time as usize {
                                let bytes = width as usize * dtype.bytes();
                                let row = (b * time as usize + t) * bytes;
                                let expected = if (offset..offset + time as usize - 1).contains(&t)
                                {
                                    original[row..row + bytes].to_vec()
                                } else {
                                    vec![0; bytes]
                                };
                                assert_eq!(restored[row..row + bytes], expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn empty_windows_and_invalid_bounds_are_checked_before_copy() {
        for shape in [[0, 5, 3], [2, 0, 3], [2, 1, 3]] {
            let source = TensorLayout::contiguous(&shape, DType::F32).unwrap();
            let plan = WindowPlan::new(
                &source,
                TokenWindow {
                    time_start: 0,
                    time_len: 0,
                    start: 0,
                    count: 0,
                },
            )
            .unwrap();
            assert_eq!(plan.spans().count(), 0);
            assert_eq!(plan.output.byte_len(), 0);
        }
        let source = TensorLayout::contiguous(&[2, 5, 3], DType::F32).unwrap();
        for window in [
            TokenWindow {
                time_start: 2,
                time_len: 4,
                start: 0,
                count: 1,
            },
            TokenWindow {
                time_start: 0,
                time_len: 4,
                start: 8,
                count: 1,
            },
            TokenWindow {
                time_start: 0,
                time_len: 0,
                start: 0,
                count: 1,
            },
            TokenWindow {
                time_start: usize::MAX,
                time_len: 1,
                start: 0,
                count: 0,
            },
            TokenWindow {
                time_start: 0,
                time_len: 4,
                start: usize::MAX,
                count: 1,
            },
        ] {
            assert!(WindowPlan::new(&source, window).is_err());
        }
        assert!(
            WindowPlan::new(
                &TensorLayout::contiguous(&[2, 5, 0], DType::F32).unwrap(),
                TokenWindow {
                    time_start: 0,
                    time_len: 4,
                    start: 0,
                    count: 1
                }
            )
            .is_err()
        );
    }
}
