mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

use std::cell::RefCell;

thread_local! {
    static BUFFER_POOL: RefCell<Vec<Vec<f32>>> = RefCell::new(Vec::new());
}

#[inline(always)]
pub fn get_buffer(size: usize) -> Vec<f32> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for i in 0..pool.len() {
            if pool[i].capacity() >= size {
                let mut buf = pool.swap_remove(i);
                buf.resize(size, 0.0);
                return buf;
            }
        }
        vec![0.0f32; size]
    })
}

#[inline(always)]
pub fn return_buffer(buf: Vec<f32>) {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 16 && !buf.is_empty() {
            pool.push(buf);
        }
    })
}

/// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
/// Cosine Functions" by Charalampidis [2016].
pub struct RecursiveGaussian;

impl RecursiveGaussian {
    #[cfg(feature = "rayon")]
    #[inline(always)]
    pub fn horizontal_pass(&self, input: &[f32], output: &mut [f32], width: usize) {
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        use rayon::prelude::ParallelSliceMut;
        use rayon::slice::ParallelSlice;

        assert_eq!(input.len(), output.len());

        input
            .par_chunks_exact(width)
            .zip(output.par_chunks_exact_mut(width))
            .for_each(|(input, output)| self.horizontal_row(input, output, width));
    }

    #[cfg(not(feature = "rayon"))]
    #[inline(always)]
    pub fn horizontal_pass(&self, input: &[f32], output: &mut [f32], width: usize) {
        assert_eq!(input.len(), output.len());

        for (input, output) in input
            .chunks_exact(width)
            .zip(output.chunks_exact_mut(width))
        {
            self.horizontal_row(input, output, width);
        }
    }

    #[inline(always)]
    fn horizontal_row(&self, input: &[f32], output: &mut [f32], width: usize) {
        let big_n = consts::RADIUS as isize;

        // consts caching
        let mul_in_1 = consts::MUL_IN_1;
        let mul_in_3 = consts::MUL_IN_3;
        let mul_in_5 = consts::MUL_IN_5;
        let mul_prev_1 = consts::MUL_PREV_1;
        let mul_prev_3 = consts::MUL_PREV_3;
        let mul_prev_5 = consts::MUL_PREV_5;
        let mul_prev2_1 = consts::MUL_PREV2_1;
        let mul_prev2_3 = consts::MUL_PREV2_3;
        let mul_prev2_5 = consts::MUL_PREV2_5;

        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        let mut n = (-big_n) + 1;
        while n < width as isize {
            let left = n - big_n - 1;
            let right = n + big_n - 1;

            let left_val = if left >= 0 {
                // SAFETY: `left` can never be bigger than `width`
                unsafe { *input.get_unchecked(left as usize) }
            } else {
                0f32
            };

            let right_val = if right < width as isize {
                // SAFETY: this branch ensures that `right` is not bigger than `width`
                unsafe { *input.get_unchecked(right as usize) }
            } else {
                0f32
            };

            let sum = left_val + right_val;

            let mut out_1 = sum * mul_in_1;
            let mut out_3 = sum * mul_in_3;
            let mut out_5 = sum * mul_in_5;

            out_1 = mul_prev2_1.mul_add(prev2_1, out_1);
            out_3 = mul_prev2_3.mul_add(prev2_3, out_3);
            out_5 = mul_prev2_5.mul_add(prev2_5, out_5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;

            out_1 = mul_prev_1.mul_add(prev_1, out_1);
            out_3 = mul_prev_3.mul_add(prev_3, out_3);
            out_5 = mul_prev_5.mul_add(prev_5, out_5);

            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if n >= 0 {
                // SAFETY: We know that this chunk of output is of size `width`,
                // which `n` cannot be larger than.
                unsafe {
                    *output.get_unchecked_mut(n as usize) = out_1 + out_3 + out_5;
                }
            }

            n += 1;
        }
    }

    #[inline(always)]
    fn transpose(&self, input: &[f32], output: &mut [f32], width: usize, height: usize) {
        assert_eq!(input.len(), width * height);
        assert_eq!(output.len(), width * height);

        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .for_each(|(x, y)| output[x * height + y] = input[y * width + x]);
    }

    #[inline(always)]
    pub fn vertical_pass(&self, input: &[f32], output: &mut [f32], width: usize, height: usize) {
        let size = width * height;
        assert_eq!(input.len(), size);
        assert_eq!(output.len(), size);

        let mut transposed_input = get_buffer(size);
        let mut transposed_output = get_buffer(size);

        // "Transpose the input data (rows <-> columns)"
        self.transpose(input, &mut transposed_input, width, height);

        #[cfg(feature = "rayon")]
        {
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            use rayon::prelude::*;

            transposed_input
                .par_chunks(height)
                .zip(transposed_output.par_chunks_mut(height))
                .for_each(|(input_chunk, output_chunk)| {
                    self.horizontal_row(input_chunk, output_chunk, height);
                });
        }

        #[cfg(not(feature = "rayon"))]
        {
            for y in 0..width {
                let start = y * height;
                let end = start + height;
                self.horizontal_row(
                    &transposed_input[start..end],
                    &mut transposed_output[start..end],
                    height,
                );
            }
        }

        self.transpose(&transposed_output, output, height, width);

        return_buffer(transposed_input);
        return_buffer(transposed_output);
    }
}
