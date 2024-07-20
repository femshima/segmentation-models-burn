# segmentation-models-burn

Rust library for image segmentation using [Burn](https://crates.io/crates/burn).

## Acknowledgements

The overall architecture and concept was inspired from [qubvel-org/segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

For each encoder/decoder,
- EfficientNet encoder is based on [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).
- ResNet encoder was copied from [tracel-ai/models](https://github.com/tracel-ai/models).
- Unet++ decoder is based on [qubvel-org/segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

## License

This library is distributed under MIT license, as in LICENSE file.

ResNet encoder code partially belongs to Burn-rs/Models Contributors. Please see [tracel-ai/models LICENSE-MIT](https://github.com/tracel-ai/models/blob/main/LICENSE-MIT).
