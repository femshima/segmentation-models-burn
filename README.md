# segmentation-models-burn

Rust library for image segmentation using [Burn](https://crates.io/crates/burn).

## Acknowledgements

The overall architecture and concept was inspired from [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

For each encoder/decoder,
- EfficientNet encoder is based on [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).
- ResNet encoder was copied from [resnet-burn](https://github.com/laggui/resnet-burn).
- Unet++ decoder is based on [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

## License

This library is distributed under MIT license, as in LICENSE file.

ResNet encoder code partially belongs to Guillaume Lagrange. Please see [resnet-burn/LICENSE](https://github.com/laggui/resnet-burn/blob/main/LICENSE).
