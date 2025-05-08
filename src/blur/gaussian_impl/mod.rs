mod gaussian;
use super::{BlurOperator, Ssimulacra2Error};

pub struct GaussianBlur {
    width: usize,
    height: usize,
    temp_buffer: Option<Vec<f32>>,
    transposed_input: Option<Vec<f32>>,
    transposed_output: Option<Vec<f32>>,
}

impl BlurOperator for GaussianBlur {
    fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        GaussianBlur {
            width,
            height,
            temp_buffer: Some(vec![0.0f32; size]),
            transposed_input: Some(vec![0.0f32; size]),
            transposed_output: Some(vec![0.0f32; size]),
        }
    }

    fn shrink_to(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;

        if let Some(buffer) = &mut self.temp_buffer {
            buffer.resize(width * height, 0.0);
        }

        if let Some(buffer) = &mut self.transposed_input {
            buffer.resize(width * height, 0.0);
        }

        if let Some(buffer) = &mut self.transposed_output {
            buffer.resize(width * height, 0.0);
        }
    }

    fn blur(
        &mut self,
        img: &[Vec<f32>; 3],
        out: &mut [Vec<f32>; 3],
    ) -> Result<(), Ssimulacra2Error> {
        let mut temp = self.temp_buffer.take().unwrap();
        let kernel = gaussian::RecursiveGaussian;

        for i in 0..3 {
            kernel.horizontal_pass(&img[i], &mut temp, self.width);
            kernel.vertical_pass(
                &temp,
                &mut out[i],
                self.width,
                self.height,
                self.transposed_input.as_mut().unwrap(),
                self.transposed_output.as_mut().unwrap(),
            );
        }

        self.temp_buffer = Some(temp);

        Ok(())
    }
}
