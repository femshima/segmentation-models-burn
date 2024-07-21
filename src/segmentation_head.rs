use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

use crate::activation::Activation;

#[derive(Module, Debug)]
pub struct SegmentationHead<B: Backend> {
    conv: Conv2d<B>,
    activation: Activation,
}

impl<B: Backend> SegmentationHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        self.activation.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct SegmentationHeadConfig {
    #[config(default = "[3, 3]")]
    kernel_size: [usize; 2],
    #[config(default = "Activation::Sigmoid")]
    activation: Activation,
}

impl SegmentationHeadConfig {
    pub fn init<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &B::Device,
    ) -> SegmentationHead<B> {
        SegmentationHead {
            conv: Conv2dConfig::new([in_channels, out_channels], self.kernel_size)
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            activation: self.activation,
        }
    }
}
