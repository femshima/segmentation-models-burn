use core::f64::consts::SQRT_2;

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, Relu,
    },
    tensor::{backend::Backend, Device, Tensor},
};

#[derive(Module, Debug)]
pub enum ResidualBlock<B: Backend> {
    /// A bottleneck residual block.
    Bottleneck(Bottleneck<B>),
    /// A basic residual block.
    Basic(BasicBlock<B>),
}

impl<B: Backend> ResidualBlock<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Basic(block) => block.forward(input),
            Self::Bottleneck(block) => block.forward(input),
        }
    }
}

#[derive(Config)]
struct ResidualBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    bottleneck: bool,
}

impl ResidualBlockConfig {
    fn init<B: Backend>(&self, device: &Device<B>) -> ResidualBlock<B> {
        if self.bottleneck {
            ResidualBlock::Bottleneck(
                BottleneckConfig::new(self.in_channels, self.out_channels, self.stride)
                    .init(device),
            )
        } else {
            ResidualBlock::Basic(
                BasicBlockConfig::new(self.in_channels, self.out_channels, self.stride)
                    .init(device),
            )
        }
    }
}

/// ResNet [basic residual block](https://paperswithcode.com/method/residual-block) implementation.
/// Derived from [torchivision.models.resnet.BasicBlock](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> BasicBlock<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Skip connection
        let out = {
            match &self.downsample {
                Some(downsample) => out + downsample.forward(identity),
                None => out + identity,
            }
        };

        // Activation
        self.relu.forward(out)
    }
}

/// ResNet [bottleneck residual block](https://paperswithcode.com/method/bottleneck-residual-block)
/// implementation.
/// Derived from [torchivision.models.resnet.Bottleneck](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
///
/// **NOTE:**  Following common practice, this bottleneck block places the stride for downsampling
/// to the second 3x3 convolution while the original paper places it to the first 1x1 convolution.
/// This variant improves the accuracy and is known as [ResNet V1.5](https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch).
#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> Bottleneck<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv3.forward(out);
        let out = self.bn3.forward(out);

        // Skip connection
        let out = {
            match &self.downsample {
                Some(downsample) => out + downsample.forward(identity),
                None => out + identity,
            }
        };

        // Activation
        self.relu.forward(out)
    }
}

/// Downsample layer applies a 1x1 conv to reduce the resolution (H, W) and adjust the number of channels.
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> Downsample<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        self.bn.forward(out)
    }
}

/// Collection of sequential residual blocks.
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    blocks: Vec<ResidualBlock<B>>,
}

impl<B: Backend> LayerBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = input;
        for block in &self.blocks {
            out = block.forward(out);
        }
        out
    }
}

/// [Basic residual block](BasicBlock) configuration.
struct BasicBlockConfig {
    conv1: Conv2dConfig,
    bn1: BatchNormConfig,
    conv2: Conv2dConfig,
    bn2: BatchNormConfig,
    downsample: Option<DownsampleConfig>,
}

impl BasicBlockConfig {
    /// Create a new instance of the residual block [config](BasicBlockConfig).
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        // conv3x3
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        let bn1 = BatchNormConfig::new(out_channels);

        // conv3x3
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        let bn2 = BatchNormConfig::new(out_channels);

        let downsample = {
            if in_channels != out_channels {
                Some(DownsampleConfig::new(in_channels, out_channels, stride))
            } else {
                None
            }
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        }
    }

    /// Initialize a new [basic residual block](BasicBlock) module.
    fn init<B: Backend>(&self, device: &Device<B>) -> BasicBlock<B> {
        // Conv initializer
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2, // recommended value for ReLU
            fan_out_only: true,
        };

        BasicBlock {
            conv1: self
                .conv1
                .clone()
                .with_initializer(initializer.clone())
                .init(device),
            bn1: self.bn1.init(device),
            relu: Relu::new(),
            conv2: self
                .conv2
                .clone()
                .with_initializer(initializer)
                .init(device),
            bn2: self.bn2.init(device),
            downsample: self.downsample.as_ref().map(|d| d.init(device)),
        }
    }
}

/// [Bottleneck residual block](Bottleneck) configuration.
struct BottleneckConfig {
    conv1: Conv2dConfig,
    bn1: BatchNormConfig,
    conv2: Conv2dConfig,
    bn2: BatchNormConfig,
    conv3: Conv2dConfig,
    bn3: BatchNormConfig,
    downsample: Option<DownsampleConfig>,
}

impl BottleneckConfig {
    /// Create a new instance of the bottleneck residual block [config](BottleneckConfig).
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        // Intermediate output channels w/ expansion = 4
        let int_out_channels = out_channels / 4;
        // conv1x1
        let conv1 = Conv2dConfig::new([in_channels, int_out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false);
        let bn1 = BatchNormConfig::new(int_out_channels);
        // conv3x3
        let conv2 = Conv2dConfig::new([int_out_channels, int_out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        let bn2 = BatchNormConfig::new(int_out_channels);
        // conv1x1
        let conv3 = Conv2dConfig::new([int_out_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false);
        let bn3 = BatchNormConfig::new(out_channels);

        let downsample = {
            if in_channels != out_channels {
                Some(DownsampleConfig::new(in_channels, out_channels, stride))
            } else {
                None
            }
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
        }
    }

    /// Initialize a new [bottleneck residual block](Bottleneck) module.
    fn init<B: Backend>(&self, device: &Device<B>) -> Bottleneck<B> {
        // Conv initializer
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2, // recommended value for ReLU
            fan_out_only: true,
        };

        Bottleneck {
            conv1: self
                .conv1
                .clone()
                .with_initializer(initializer.clone())
                .init(device),
            bn1: self.bn1.init(device),
            relu: Relu::new(),
            conv2: self
                .conv2
                .clone()
                .with_initializer(initializer.clone())
                .init(device),
            bn2: self.bn2.init(device),
            conv3: self
                .conv3
                .clone()
                .with_initializer(initializer)
                .init(device),
            bn3: self.bn3.init(device),
            downsample: self.downsample.as_ref().map(|d| d.init(device)),
        }
    }
}

/// [Downsample](Downsample) configuration.
struct DownsampleConfig {
    conv: Conv2dConfig,
    bn: BatchNormConfig,
}

impl DownsampleConfig {
    /// Create a new instance of the downsample [config](DownsampleConfig).
    fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        // conv1x1
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false);
        let bn = BatchNormConfig::new(out_channels);

        Self { conv, bn }
    }

    /// Initialize a new [downsample](Downsample) module.
    fn init<B: Backend>(&self, device: &B::Device) -> Downsample<B> {
        // Conv initializer
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2, // recommended value for ReLU
            fan_out_only: true,
        };

        Downsample {
            conv: self.conv.clone().with_initializer(initializer).init(device),
            bn: self.bn.init(device),
        }
    }
}

/// [Residual layer block](LayerBlock) configuration.
#[derive(Config)]
pub struct LayerBlockConfig {
    num_blocks: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    bottleneck: bool,
}

impl LayerBlockConfig {
    /// Initialize a new [LayerBlock](LayerBlock) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> LayerBlock<B> {
        let blocks = (0..self.num_blocks)
            .map(|b| {
                if b == 0 {
                    // First block uses the specified stride
                    ResidualBlockConfig::new(
                        self.in_channels,
                        self.out_channels,
                        self.stride,
                        self.bottleneck,
                    )
                    .init(device)
                } else {
                    // Other blocks use a stride of 1
                    ResidualBlockConfig::new(
                        self.out_channels,
                        self.out_channels,
                        1,
                        self.bottleneck,
                    )
                    .init(device)
                }
            })
            .collect();

        LayerBlock { blocks }
    }
}
