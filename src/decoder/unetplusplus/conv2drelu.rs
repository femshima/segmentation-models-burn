use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Conv2dReLU<B: Backend> {
    conv: Conv2d<B>,
    batchnorm: Option<BatchNorm<B, 2>>,
    relu: Relu,
}

impl<B: Backend> Conv2dReLU<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.conv.forward(x);

        if let Some(ref batchnorm) = self.batchnorm {
            x = batchnorm.forward(x);
        }

        self.relu.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct Conv2dReLUConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
    padding: PaddingConfig2d,
    stride: [usize; 2],
    use_batchnorm: bool,
}

impl Conv2dReLUConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2dReLU<B> {
        Conv2dReLU {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], self.kernel_size)
                .with_stride(self.stride)
                .with_padding(self.padding.clone())
                .with_bias(!self.use_batchnorm)
                .init(device),
            batchnorm: if self.use_batchnorm {
                Some(BatchNormConfig::new(self.out_channels).init(device))
            } else {
                None
            },
            relu: Relu::new(),
        }
    }
}
