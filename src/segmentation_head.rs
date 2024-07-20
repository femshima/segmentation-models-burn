use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    tensor::{activation::sigmoid, backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct SegmentationHead<B: Backend> {
    conv: Conv2d<B>,
    omit_activation_on_train: bool,
}

impl<B: Backend> SegmentationHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.conv.forward(x);
        if !self.omit_activation_on_train && B::ad_enabled() {
            x = sigmoid(x);
        }
        x
    }
}

#[derive(Config, Debug)]
pub struct SegmentationHeadConfig {
    in_channels: usize,
    out_channels: usize,
    #[config(default = "[3, 3]")]
    kernel_size: [usize; 2],
    #[config(default = false)]
    omit_activation_on_train: bool,
}

impl SegmentationHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SegmentationHead<B> {
        SegmentationHead {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], self.kernel_size)
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            omit_activation_on_train: self.omit_activation_on_train,
        }
    }
}
