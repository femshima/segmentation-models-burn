pub mod unetplusplus;

use burn::tensor::{backend::Backend, Tensor};

pub trait Decoder {
    type Backend: Backend;
    fn forward(&self, x: Vec<Tensor<Self::Backend, 4>>) -> Tensor<Self::Backend, 4>;
}

pub trait DecoderConfig<B: Backend> {
    type Decoder: Decoder;
    fn init(&self, device: &B::Device) -> Self::Decoder;
    fn out_channels(&self) -> usize;
    fn with_encoder_channels(&self, encoder_channels: Vec<usize>) -> Self;
}
