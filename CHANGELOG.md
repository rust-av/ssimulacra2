## Version 0.5.0

- Return a concrete `Ssimulacra2Error` error type instead of a freeform `anyhow::Result`
- Precalculate float consts for RecursiveGaussian at build time (performance)
- Update `yuvxyb` dependency to 0.4

## Version 0.4.0

- Update to [version 2.1 of the metric](https://github.com/cloudinary/ssimulacra2/compare/v2.0...v2.1)

## Version 0.3.1

- Minor optimizations
- Bump `nalgebra` dependency to 0.32

## Version 0.3.0

- [Breaking] Reexported structs from yuvxyb have had `From<&T>` impls removed
- Considerably speedups and optimizations

## Version 0.2.0

- [Breaking] Implement updates to the algorithm from upstream (https://github.com/libjxl/libjxl/pull/1848)
- Bump yuvxyb version
- Speed improvements

## Version 0.1.0

- Initial release
