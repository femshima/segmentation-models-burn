use burn::{
    nn::PaddingConfig2d,
    prelude::*,
    tensor::ops::{InterpolateMode, InterpolateOptions},
};

use crate::decoder::{Decoder, DecoderConfig};

use super::conv2drelu::{Conv2dReLU, Conv2dReLUConfig};

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    conv1: Conv2dReLU<B>,
    conv2: Conv2dReLU<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>, skip: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
        let shape = x.shape();
        let x = burn::tensor::module::interpolate(
            x,
            [shape.dims[2] * 2, shape.dims[3] * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );

        let x = if let Some(skip) = skip {
            Tensor::cat(vec![x, skip], 1)
        } else {
            x
        };

        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);

        x
    }
}

#[derive(Config, Debug)]
pub struct DecoderBlockConfig {
    in_channels: usize,
    skip_channels: usize,
    out_channels: usize,
    use_batchnorm: bool,
}

impl DecoderBlockConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        DecoderBlock {
            conv1: Conv2dReLUConfig::new(
                self.in_channels + self.skip_channels,
                self.out_channels,
                [3, 3],
                PaddingConfig2d::Explicit(1, 1),
                [1, 1],
                self.use_batchnorm,
            )
            .init(device),
            conv2: Conv2dReLUConfig::new(
                self.out_channels,
                self.out_channels,
                [3, 3],
                PaddingConfig2d::Explicit(1, 1),
                [1, 1],
                self.use_batchnorm,
            )
            .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct UnetPlusPlusDecoder<B: Backend> {
    blocks: Vec<Vec<DecoderBlock<B>>>,
    output_block: DecoderBlock<B>,
}

impl<B: Backend> Decoder for UnetPlusPlusDecoder<B> {
    type Backend = B;
    fn forward(&self, features: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        let mut feature_iter = features.into_iter().rev();

        let mut ins = Vec::with_capacity(self.blocks.len());
        let mut skips = Vec::with_capacity(self.blocks.len());

        ins.push(feature_iter.next().unwrap());

        for layer in self.blocks.iter().rev() {
            skips.push(feature_iter.next().unwrap());

            for (block, in_ch) in layer.iter().rev().zip(&ins) {
                let output = block.forward(in_ch.clone(), Some(Tensor::cat(skips.clone(), 1)));
                skips.push(output);
            }

            std::mem::swap(&mut ins, &mut skips);
            skips.clear();
        }

        self.output_block.forward(ins.last().unwrap().clone(), None)
    }
}

#[derive(Config, Debug)]
pub struct UnetPlusPlusDecoderConfig {
    encoder_channels: Vec<usize>,
    decoder_channels: Vec<usize>,

    #[config(default = true)]
    use_batchnorm: bool,
}

impl<B: Backend> DecoderConfig<B> for UnetPlusPlusDecoderConfig {
    type Decoder = UnetPlusPlusDecoder<B>;
    fn out_channels(&self) -> usize {
        *self.decoder_channels.last().unwrap()
    }
    fn with_encoder_channels(&self, encoder_channels: Vec<usize>) -> Self {
        let mut s = self.clone();
        s.encoder_channels = encoder_channels;
        s
    }
    fn init(&self, device: &B::Device) -> UnetPlusPlusDecoder<B> {
        let main_channel_iter = std::iter::once([
            self.encoder_channels.last().unwrap(),
            self.decoder_channels.first().unwrap(),
        ])
        .chain(self.decoder_channels.windows(2).map(|v| match v {
            [in_ch, out_ch] => [in_ch, out_ch],
            _ => unreachable!(),
        }))
        .rev()
        .skip(1);

        let skip_channel_iter = self.encoder_channels.windows(2).skip(1).map(|v| match v {
            [out_ch, in_ch] => [in_ch, out_ch],
            _ => unreachable!(),
        });

        let blocks = main_channel_iter
            .enumerate()
            .zip(skip_channel_iter)
            .map(|((i, [main_in, main_out]), [skip_in, skip_out])| {
                let skip_count_max = self.decoder_channels.len() - i - 1;

                let main_block = std::iter::once(
                    DecoderBlockConfig::new(
                        *main_in,
                        skip_out * skip_count_max,
                        *main_out,
                        self.use_batchnorm,
                    )
                    .init(device),
                );

                let sub_blocks = (1..skip_count_max).rev().map(|skip_count| {
                    DecoderBlockConfig::new(
                        *skip_in,
                        skip_out * skip_count,
                        *skip_out,
                        self.use_batchnorm,
                    )
                    .init(device)
                });

                main_block.chain(sub_blocks).collect()
            })
            .collect();

        let output_block = DecoderBlockConfig::new(
            self.decoder_channels[self.decoder_channels.len() - 2],
            0,
            self.decoder_channels[self.decoder_channels.len() - 1],
            self.use_batchnorm,
        )
        .init(device);

        UnetPlusPlusDecoder {
            blocks,
            output_block,
        }
    }
}
