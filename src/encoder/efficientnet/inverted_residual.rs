use burn::config::Config;
use burn::nn::conv::Conv2dConfig;
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig};
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::module::adaptive_avg_pool2d;
use burn::tensor::Tensor;
use burn::{module::Module, nn::conv::Conv2d, tensor::backend::Backend};

use super::conv_norm::{Conv2dNormActivation, Conv2dNormActivationConfig};

#[derive(Module, Debug)]
pub struct PointWiseLinear<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
}

impl<B: Backend> PointWiseLinear<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.norm.forward(self.conv.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct SqueezeAndExcitation<B: Backend> {
    reduce: Conv2d<B>,
    expand: Conv2d<B>,
}

impl<B: Backend> SqueezeAndExcitation<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x_squeezed = adaptive_avg_pool2d(x.clone(), [1, 1]);
        let x_squeezed = self.reduce.forward(x_squeezed);
        let x_squeezed = silu(x_squeezed);
        let x_squeezed = self.expand.forward(x_squeezed);
        sigmoid(x_squeezed) * x
    }
}

/// [Inverted Residual Block](https://paperswithcode.com/method/inverted-residual-block).
#[derive(Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    use_res_connect: bool,
    pw: Option<Conv2dNormActivation<B>>, // pointwise, only when expand ratio != 1
    dw: Conv2dNormActivation<B>,
    pw_linear: PointWiseLinear<B>,
    se: Option<SqueezeAndExcitation<B>>,

    dropout: Option<Dropout>,
}

impl<B: Backend> InvertedResidual<B> {
    pub fn forward(&self, x: &Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = x.clone();
        if let Some(pw) = &self.pw {
            out = pw.forward(out);
        }

        out = self.dw.forward(out);
        if let Some(se) = &self.se {
            out = se.forward(out);
        }

        out = self.pw_linear.forward(out);

        if self.use_res_connect {
            if let Some(dropout) = &self.dropout {
                out = dropout.forward(out);
            }
            out = out + x.clone();
        }
        out
    }
}

/// [InvertedResidual](InvertedResidual) configuration.
#[derive(Config, Debug)]
pub struct InvertedResidualConfig {
    pub num_repeat: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub expand_ratio: usize,
    pub input_filters: usize,
    pub output_filters: usize,
    pub se_ratio: Option<f64>,
    pub id_skip: bool,

    pub drop_connect_rate: Option<f64>,
}

impl InvertedResidualConfig {
    /// Initialize a new [InvertedResidual](InvertedResidual) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidual<B> {
        let hidden_dim = self.input_filters * self.expand_ratio;
        let pw = if self.expand_ratio != 1 {
            Some(
                Conv2dNormActivationConfig::new(self.input_filters, hidden_dim)
                    .with_kernel_size(1)
                    .with_bias(false)
                    .init(device),
            )
        } else {
            None
        };
        let dw = Conv2dNormActivationConfig::new(hidden_dim, hidden_dim)
            .with_stride(self.stride)
            .with_kernel_size(self.kernel_size)
            .with_groups(hidden_dim)
            .with_bias(false)
            .init(device);
        let pw_linear = PointWiseLinear {
            conv: Conv2dConfig::new([hidden_dim, self.output_filters], [1, 1])
                .with_stride([1, 1])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(0, 0))
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(self.output_filters).init(device),
        };
        let se = if let Some(se_ratio) = self.se_ratio {
            assert!(0.0 < se_ratio && se_ratio <= 1.0);
            let num_squeezed_channels = ((self.input_filters as f64 * se_ratio) as usize).max(1);
            Some(SqueezeAndExcitation {
                reduce: Conv2dConfig::new([hidden_dim, num_squeezed_channels], [1, 1])
                    .with_padding(burn::nn::PaddingConfig2d::Same)
                    .init(device),
                expand: Conv2dConfig::new([num_squeezed_channels, hidden_dim], [1, 1])
                    .with_padding(burn::nn::PaddingConfig2d::Same)
                    .init(device),
            })
        } else {
            None
        };
        InvertedResidual {
            use_res_connect: self.id_skip
                && self.stride == 1
                && self.input_filters == self.output_filters,
            pw_linear,
            dw,
            pw,
            se,

            dropout: self
                .drop_connect_rate
                .map(|prob| DropoutConfig::new(prob).init()),
        }
    }

    pub fn encode(&self) -> String {
        let se = if let Some(se_ratio) = self.se_ratio {
            assert!(se_ratio > 0.0);
            assert!(se_ratio <= 1.0);
            format!("_se{}", se_ratio)
        } else {
            "".to_string()
        };
        let skip = if self.id_skip { "" } else { "_noskip" };

        format!(
            "r{}_k{}_s{}{}_e{}_i{}_o{}{}{}",
            self.num_repeat,
            self.kernel_size,
            self.stride,
            self.stride,
            self.expand_ratio,
            self.input_filters,
            self.output_filters,
            se,
            skip
        )
    }
    pub fn decode(s: &str) -> Self {
        let mut map = std::collections::HashMap::with_capacity(8);

        for param in s.split('_') {
            if param.is_empty() {
                continue;
            }

            let (mut k, mut v) = param.split_at(1);
            if param.starts_with("se") {
                k = "se";
                v = &v[1..];
            }
            if param == "noskip" {
                k = "noskip";
                v = "";
            }
            if k == "s" {
                match v.len() {
                    1 => (),
                    2 => {
                        assert_eq!(v[0..1], v[1..2]);
                        v = &v[0..1];
                    }
                    len => panic!(
                        "stride string's length must be 1 or 2, instead got {:?}",
                        len
                    ),
                }
            }
            map.insert(k, v);
        }

        Self {
            num_repeat: map.get("r").unwrap().parse().unwrap(),
            kernel_size: map.get("k").unwrap().parse().unwrap(),
            stride: map.get("s").unwrap().parse().unwrap(),
            expand_ratio: map.get("e").unwrap().parse().unwrap(),
            input_filters: map.get("i").unwrap().parse().unwrap(),
            output_filters: map.get("o").unwrap().parse().unwrap(),

            se_ratio: map.get("se").map(|v| v.parse().unwrap()),
            id_skip: map.contains_key("noskip"),

            drop_connect_rate: None,
        }
    }
}
