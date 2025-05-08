#[cfg(not(feature = "libblur"))]
mod gaussian_impl;
#[cfg(all(feature = "rayon", feature = "libblur"))]
mod libblur_impl;
use crate::Ssimulacra2Error;

/// Trait handling image blur.
///
/// This trait contains the necessary buffers and the kernel used for blurring
/// (currently a recursive approximation of the Gaussian filter).
///
/// Note that the width and height of the image passed to [blur][Self::blur] needs to exactly
/// match the width and height of this instance. If you reduce the image size (e.g. via
/// downscaling), [`shrink_to`][Self::shrink_to] can be used to resize the internal buffers.
pub trait BlurOperator {
    /// Create a new [Blur] for images of the given width and height.
    fn new(width: usize, height: usize) -> Self;
    /// Truncates the internal buffers to fit images of the given width and height.
    fn shrink_to(&mut self, width: usize, height: usize);
    /// Blur the given image using libblur's gaussian_blur_f32.
    fn blur(
        &mut self,
        img: &[Vec<f32>; 3],
        out: &mut [Vec<f32>; 3],
    ) -> Result<(), Ssimulacra2Error>;
}

#[cfg(not(feature = "libblur"))]
pub use gaussian_impl::GaussianBlur as Blur;

#[cfg(all(feature = "rayon", feature = "libblur"))]
pub use libblur_impl::LibBlur as Blur;
