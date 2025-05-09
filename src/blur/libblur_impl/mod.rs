use super::{BlurOperator, Ssimulacra2Error};
use libblur::{BlurImage, BlurImageMut, EdgeMode, FastBlurChannels, ThreadingPolicy};
use std::borrow::Cow;

pub struct LibBlur {
    width: usize,
    height: usize,
}

impl BlurOperator for LibBlur {
    fn new(width: usize, height: usize) -> Self {
        LibBlur { width, height }
    }

    fn shrink_to(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
    }

    fn blur(
        &mut self,
        img: &[Vec<f32>; 3],
        out: &mut [Vec<f32>; 3],
    ) -> Result<(), Ssimulacra2Error> {
        self.blur_plane(&img[0], &mut out[0])?;
        self.blur_plane(&img[1], &mut out[1])?;
        self.blur_plane(&img[2], &mut out[2])?;
        Ok(())
    }
}

impl LibBlur {
    fn blur_plane(&mut self, plane: &[f32], out: &mut [f32]) -> Result<(), Ssimulacra2Error> {
        //Set kernel size and sigma value - adjust as needed but not recommended to change
        const KERNEL_SIZE: u32 = 11;
        const SIGMA: f32 = 2.2943; // diff 0.000 / 0.932

        // BlurImage creation
        let src_image = BlurImage {
            data: Cow::Borrowed(plane),
            width: self.width as u32,
            height: self.height as u32,
            stride: self.width as u32, // stride == width
            channels: FastBlurChannels::Plane,
        };

        // BlurImageMut creation
        let mut dst_image = BlurImageMut::borrow(
            out,
            self.width as u32,
            self.height as u32,
            FastBlurChannels::Plane,
        );

        // gaussian_blur_f32 call
        #[cfg(feature = "rayon")]
        if let Err(e) = libblur::gaussian_blur_f32(
            &src_image,
            &mut dst_image,
            KERNEL_SIZE,
            SIGMA,
            EdgeMode::Clamp,
            ThreadingPolicy::Adaptive, // use rayon
        ) {
            eprintln!("Error in gaussian_blur_f32: {:?}", e);
            return Err(Ssimulacra2Error::GaussianBlurError);
        }

        #[cfg(not(feature = "rayon"))]
        if let Err(e) = libblur::gaussian_blur_f32(
            &src_image,
            &mut dst_image,
            KERNEL_SIZE,
            SIGMA,
            EdgeMode::Clamp,
            ThreadingPolicy::Single, // no rayon
        ) {
            eprintln!("Error in gaussian_blur_f32: {:?}", e);
            return Err(Ssimulacra2Error::GaussianBlurError);
        }

        Ok(())
    }
}
