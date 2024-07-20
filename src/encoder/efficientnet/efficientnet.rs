use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d,
    },
    tensor::{activation::silu, backend::Backend, Device, Tensor},
};

use crate::encoder::{Encoder, EncoderConfig};

use super::inverted_residual::{InvertedResidual, InvertedResidualConfig};

#[derive(Module, Debug)]
pub struct EfficientNet<B: Backend> {
    conv_stem: Conv2d<B>,
    bn0: BatchNorm<B, 2>,
    blocks: Vec<InvertedResidual<B>>,
    conv_head: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    avg_pooling: AdaptiveAvgPool2d,
}

impl<B: Backend> Encoder for EfficientNet<B> {
    type Backend = B;
    fn forward(
        &self,
        x: Tensor<Self::Backend, 4>,
        feature_idxs: Vec<usize>,
    ) -> Vec<Tensor<Self::Backend, 4>> {
        let mut features = Vec::with_capacity(self.blocks.len());
        features.push(x.clone());

        // Stem
        let mut x = silu(self.bn0.forward(self.conv_stem.forward(x)));
        features.push(x.clone());

        // Blocks
        for (idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x);

            if feature_idxs.contains(&idx) {
                features.push(x.clone());
            }
        }

        // Head
        // silu(self.bn1.forward(self.conv_head.forward(x)));

        features
    }
}

#[derive(Debug, Config)]
pub struct EfficientNetGlobalConfig {
    width_coefficient: Option<f64>,
    depth_coefficient: Option<f64>,
    image_size: Option<usize>,
    #[config(default = 0.2)]
    dropout_rate: f64,

    #[config(default = 1000)]
    num_classes: usize,
    #[config(default = 0.90)]
    batch_norm_momentum: f64,
    #[config(default = 1e-3)]
    batch_norm_epsilon: f64,
    #[config(default = 0.2)]
    drop_connect_rate: f64,
    #[config(default = 8)]
    depth_divisor: usize,
    min_depth: Option<usize>,
    #[config(default = true)]
    include_top: bool,
}

impl EfficientNetGlobalConfig {
    fn round_filters(&self, filters: usize) -> usize {
        let Some(multiplier) = self.width_coefficient else {
            return filters;
        };

        let filters = filters as f64 * multiplier;
        let min_depth = self.min_depth.unwrap_or(self.depth_divisor);

        let mut new_filters = min_depth.max(
            ((filters + self.depth_divisor as f64 / 2.0).floor() / self.depth_divisor as f64)
                as usize
                * self.depth_divisor,
        );

        if (new_filters as f64) < 0.9 * filters {
            new_filters += self.depth_divisor;
        }

        new_filters
    }
    fn round_repeats(&self, repeats: usize) -> usize {
        let Some(multiplier) = self.depth_coefficient else {
            return repeats;
        };

        (multiplier * repeats as f64).ceil() as usize
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EfficientNetConfig {
    bn_mom: f64,
    bn_eps: f64,

    stem_out: usize,
    head_out: usize,

    blocks: Vec<InvertedResidualConfig>,
    feature_idxs: Vec<usize>,
}

impl EfficientNetConfig {
    pub fn new(
        global: &EfficientNetGlobalConfig,
        stages: &[InvertedResidualConfig],
        feature_idxs: Option<Vec<usize>>,
    ) -> Self {
        // Batch norm parameters
        let bn_mom = global.batch_norm_momentum;
        let bn_eps = global.batch_norm_epsilon;

        // Stem
        let stem_out = global.round_filters(32);

        // Blocks
        let block_repeats: Vec<usize> = stages
            .iter()
            .map(|stage| global.round_repeats(stage.num_repeat))
            .collect();
        let mut blocks: Vec<InvertedResidualConfig> = stages
            .iter()
            .zip(&block_repeats)
            .flat_map(|(stage, &num_repeat)| {
                let input_filters = global.round_filters(stage.input_filters);
                let output_filters = global.round_filters(stage.output_filters);

                (0..num_repeat).map(move |layer_idx| {
                    let mut block = stage.clone();

                    block.input_filters = if layer_idx == 0 {
                        input_filters
                    } else {
                        output_filters
                    };
                    block.output_filters = output_filters;

                    if layer_idx > 0 {
                        block.stride = 1;
                    }

                    block
                })
            })
            .collect();
        let block_len = blocks.len() as f64;
        blocks
            .iter_mut()
            .enumerate()
            .for_each(|(block_idx, block)| {
                block.drop_connect_rate = Some(global.drop_connect_rate * block_idx as f64 / block_len)
            });

        let feature_idxs = feature_idxs.unwrap_or_else(|| {
            let cumulative_num_block: Vec<usize> = block_repeats
                .iter()
                .scan(0, |acc, curr| {
                    *acc += curr;
                    Some(*acc)
                })
                .collect();
            vec![
                cumulative_num_block[1],
                cumulative_num_block[2],
                cumulative_num_block[4],
                cumulative_num_block[6],
            ]
        });

        // Head
        let head_out = global.round_filters(1280);

        Self {
            bn_mom,
            bn_eps,
            stem_out,
            head_out,
            blocks,

            feature_idxs,
        }
    }
}

impl<B: Backend> EncoderConfig<B> for EfficientNetConfig {
    type Encoder = EfficientNet<B>;
    fn out_channels(&self) -> Vec<usize> {
        [3, self.stem_out]
            .into_iter()
            .chain(
                self.feature_idxs
                    .iter()
                    .take(3)
                    .map(|&idx| self.blocks[idx].output_filters),
            )
            .chain([self.blocks.last().unwrap().output_filters])
            .collect()
    }
    fn feature_idxs(&self) -> Vec<usize> {
        self.feature_idxs.clone()
    }
    /// Initialize a new [`EfficientNet`] module.
    fn init(&self, device: &Device<B>) -> EfficientNet<B> {
        // Stem
        let conv_stem = Conv2dConfig::new([3, self.stem_out], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .init(device);
        let bn0 = BatchNormConfig::new(self.stem_out)
            .with_momentum(self.bn_mom)
            .with_epsilon(self.bn_eps)
            .init(device);

        // Blocks
        let mut blocks_out = 0;
        let blocks = self
            .blocks
            .iter()
            .map(|block| {
                blocks_out = block.output_filters;
                block.init(device)
            })
            .collect();

        // Head
        let conv_head = Conv2dConfig::new([blocks_out, self.head_out], [1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device);
        let bn1 = BatchNormConfig::new(self.head_out)
            .with_momentum(self.bn_mom)
            .with_epsilon(self.bn_eps)
            .init(device);

        // Final linear layer
        let avg_pooling = AdaptiveAvgPool2dConfig::new([1, 1]).init();

        // if self._global_params.include_top:
        //     self._dropout = nn.Dropout(self._global_params.dropout_rate)
        //     self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        EfficientNet {
            conv_stem,
            bn0,
            blocks,
            conv_head,
            bn1,
            avg_pooling,
        }
    }
}
