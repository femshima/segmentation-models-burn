use core::f64::consts::SQRT_2;

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    tensor::{backend::Backend, Device, Tensor},
};

use crate::encoder::Encoder;

use super::block::{LayerBlock, LayerBlockConfig};

/// ResNet implementation.
/// Derived from [torchivision.models.resnet.ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    maxpool: MaxPool2d,
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> Encoder for ResNet<B> {
    type Backend = B;
    fn forward(&self, x: Tensor<Self::Backend, 4>, _: Vec<usize>) -> Vec<Tensor<Self::Backend, 4>> {
        let mut features = Vec::with_capacity(6);
        features.push(x.clone());

        // First block
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        features.push(x.clone());
        let x = self.maxpool.forward(x);

        // Residual blocks
        let x = self.layer1.forward(x);
        features.push(x.clone());
        let x = self.layer2.forward(x);
        features.push(x.clone());
        let x = self.layer3.forward(x);
        features.push(x.clone());
        let x = self.layer4.forward(x);
        features.push(x.clone());

        features
    }
}

impl<B: Backend> ResNet<B> {
    /// Re-initialize the last layer with the specified number of output classes.
    pub fn with_classes(mut self, num_classes: usize) -> Self {
        let [d_input, _d_output] = self.fc.weight.dims();
        self.fc = LinearConfig::new(d_input, num_classes).init(&self.fc.weight.device());
        self
    }
}

/// [ResNet](ResNet) configuration.
pub struct ResNetConfig {
    conv1: Conv2dConfig,
    bn1: BatchNormConfig,
    maxpool: MaxPool2dConfig,
    layer1: LayerBlockConfig,
    layer2: LayerBlockConfig,
    layer3: LayerBlockConfig,
    layer4: LayerBlockConfig,
    avgpool: AdaptiveAvgPool2dConfig,
    fc: LinearConfig,
}

impl ResNetConfig {
    /// Create a new instance of the ResNet [config](ResNetConfig).
    pub fn new(blocks: [usize; 4], num_classes: usize, expansion: usize) -> Self {
        // `new()` is private but still check just in case...
        assert!(
            expansion == 1 || expansion == 4,
            "ResNet module only supports expansion values [1, 4] for residual blocks"
        );

        // 7x7 conv, 64, /2
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false);
        let bn1 = BatchNormConfig::new(64);

        // 3x3 maxpool, /2
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1));

        // Residual blocks
        let bottleneck = expansion > 1;
        let layer1 = LayerBlockConfig::new(blocks[0], 64, 64 * expansion, 1, bottleneck);
        let layer2 =
            LayerBlockConfig::new(blocks[1], 64 * expansion, 128 * expansion, 2, bottleneck);
        let layer3 =
            LayerBlockConfig::new(blocks[2], 128 * expansion, 256 * expansion, 2, bottleneck);
        let layer4 =
            LayerBlockConfig::new(blocks[3], 256 * expansion, 512 * expansion, 2, bottleneck);

        // Average pooling [B, 512 * expansion, H, W] -> [B, 512 * expansion, 1, 1]
        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]);

        // Output layer
        let fc = LinearConfig::new(512 * expansion, num_classes);

        Self {
            conv1,
            bn1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
        }
    }
}

impl ResNetConfig {
    /// Initialize a new [ResNet](ResNet) module.
    pub fn init<B: Backend>(self, device: &Device<B>) -> ResNet<B> {
        // Conv initializer
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2, // recommended value for ReLU
            fan_out_only: true,
        };

        ResNet {
            conv1: self.conv1.with_initializer(initializer).init(device),
            bn1: self.bn1.init(device),
            relu: Relu::new(),
            maxpool: self.maxpool.init(),
            layer1: self.layer1.init(device),
            layer2: self.layer2.init(device),
            layer3: self.layer3.init(device),
            layer4: self.layer4.init(device),
            avgpool: self.avgpool.init(),
            fc: self.fc.init(device),
        }
    }
}
