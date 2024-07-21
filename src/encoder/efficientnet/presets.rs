use std::path::{Path, PathBuf};

use burn::{
    config::Config,
    module::Module,
    record::{FullPrecisionSettings, Recorder, RecorderError},
    tensor::{backend::Backend, Device},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

use crate::encoder::EncoderConfig;

use super::{
    inverted_residual::InvertedResidualConfig, EfficientNet, EfficientNetConfig,
    EfficientNetGlobalConfig, EfficientNetRecord,
};

#[derive(Debug, Config)]
pub struct EfficientNetConfigPreset {
    structure: EfficientNetKind,
    weight: Option<EfficientNetWeightKind>,
}

impl EfficientNetConfigPreset {
    pub fn efficientnet_pretrained<B: Backend>(
        &self,
        device: &Device<B>,
    ) -> Result<EfficientNet<B>, RecorderError> {
        let weight: Option<PathBuf> = match self.weight {
            #[cfg(feature = "pretrained")]
            Some(EfficientNetWeightKind::Normal) => {
                let url = self.structure.to_url().unwrap();
                Some(crate::download("efficientnet", url).map_err(|err| {
                    RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
                })?)
            }
            #[cfg(feature = "pretrained")]
            Some(EfficientNetWeightKind::AdvProp) => {
                let url = self.structure.to_url_advprop().unwrap();
                Some(crate::download("efficientnet", url).map_err(|err| {
                    RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
                })?)
            }
            None => None,
            #[allow(unreachable_patterns)]
            _ => panic!("Please enable `pretrained` feature to use pretrained weights."),
        };
        let model = self.structure.to_config().init(device);

        if let Some(weight) = weight {
            let record = self.structure.load_weights_record(weight, device)?;
            Ok(model.load_record(record))
        } else {
            Ok(model)
        }
    }
}

impl<B: Backend> EncoderConfig<B> for EfficientNetConfigPreset {
    type Encoder = EfficientNet<B>;
    fn init(&self, device: &B::Device) -> Self::Encoder {
        self.efficientnet_pretrained(device).unwrap()
    }
    fn out_channels(&self) -> Vec<usize> {
        EncoderConfig::<B>::out_channels(&self.structure.to_config())
    }
    fn feature_idxs(&self) -> Vec<usize> {
        EncoderConfig::<B>::feature_idxs(&self.structure.to_config())
    }
}

#[derive(Debug, Config)]
pub enum EfficientNetWeightKind {
    Normal,
    AdvProp,
}

#[derive(Debug, Config)]
pub enum EfficientNetKind {
    EfficientnetB0,
    EfficientnetB1,
    EfficientnetB2,
    EfficientnetB3,
    EfficientnetB4,
    EfficientnetB5,
    EfficientnetB6,
    EfficientnetB7,
    EfficientnetB8,
    EfficientnetL2,
}

impl From<EfficientNetKind> for EfficientNetConfig {
    fn from(value: EfficientNetKind) -> Self {
        value.to_config()
    }
}

impl EfficientNetKind {
    pub fn to_config(&self) -> EfficientNetConfig {
        use EfficientNetKind::*;
        let (w, d, s, p) = match self {
            EfficientnetB0 => (1.0, 1.0, 224, 0.2),
            EfficientnetB1 => (1.0, 1.1, 240, 0.2),
            EfficientnetB2 => (1.1, 1.2, 260, 0.3),
            EfficientnetB3 => (1.2, 1.4, 300, 0.3),
            EfficientnetB4 => (1.4, 1.8, 380, 0.4),
            EfficientnetB5 => (1.6, 2.2, 456, 0.4),
            EfficientnetB6 => (1.8, 2.6, 528, 0.5),
            EfficientnetB7 => (2.0, 3.1, 600, 0.5),
            EfficientnetB8 => (2.2, 3.6, 672, 0.5),
            EfficientnetL2 => (4.3, 5.3, 800, 0.5),
        };

        let global_config = EfficientNetGlobalConfig::new()
            .with_width_coefficient(Some(w))
            .with_depth_coefficient(Some(d))
            .with_dropout_rate(p)
            .with_image_size(Some(s));

        let feature_idxs = match self {
            EfficientnetB0 => Some(vec![2, 4, 8, 15]),
            _ => None,
        };

        let default_block_args = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ];

        EfficientNetConfig::new(
            &global_config,
            &default_block_args
                .iter()
                .map(|args| InvertedResidualConfig::decode(args))
                .collect::<Vec<_>>(),
            feature_idxs,
        )
    }

    #[cfg(feature = "pretrained")]
    fn to_url(&self) -> Option<&str> {
        use EfficientNetKind::*;
        match self {
          // FIXME: these links do not work
          EfficientnetB0 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"),
          EfficientnetB1 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth"),
          EfficientnetB2 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth"),
          EfficientnetB3 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth"),
          EfficientnetB4 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth"),
          EfficientnetB5 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth"),
          EfficientnetB6 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth"),
          EfficientnetB7 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth"),
          _ => None,
      }
    }
    #[cfg(feature = "pretrained")]
    fn to_url_advprop(&self) -> Option<&str> {
        use EfficientNetKind::*;
        match self {
          // FIXME: these links do not work
          EfficientnetB0 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth"),
          EfficientnetB1 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth"),
          EfficientnetB2 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth"),
          EfficientnetB3 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth"),
          EfficientnetB4 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth"),
          EfficientnetB5 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth"),
          EfficientnetB6 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth"),
          EfficientnetB7 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth"),
          EfficientnetB8 => Some("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth"),
          _ => None,
      }
    }

    /// Load specified pre-trained PyTorch weights as a record.
    fn load_weights_record<B: Backend, P: AsRef<Path>>(
        &self,
        torch_weights: P,
        device: &Device<B>,
    ) -> Result<EfficientNetRecord<B>, RecorderError> {
        // Load weights from torch state_dict
        let load_args = LoadArgs::new(torch_weights.as_ref().into())
            .with_key_remap(
                "_blocks\\.([0-9]+)\\._expand_conv\\.(.+)",
                "blocks.$1.pw.conv.$2",
            )
            .with_key_remap("_blocks\\.([0-9]+)\\._bn0\\.(.+)", "blocks.$1.pw.norm.$2")
            .with_key_remap(
                "_blocks\\.([0-9]+)\\._depthwise_conv\\.(.+)",
                "blocks.$1.dw.conv.$2",
            )
            .with_key_remap("_blocks\\.([0-9]+)\\._bn1\\.(.+)", "blocks.$1.dw.norm.$2")
            .with_key_remap(
                "_blocks\\.([0-9]+)\\._project_conv\\.(.+)",
                "blocks.$1.pw_linear.conv.$2",
            )
            .with_key_remap(
                "_blocks\\.([0-9]+)\\._bn2\\.(.+)",
                "blocks.$1.pw_linear.norm.$2",
            )
            .with_key_remap("_blocks\\.([0-9]+)\\._se_(.+)", "blocks.$1.se.$2")
            .with_key_remap("_blocks\\.([0-9]+)\\._(.+)", "blocks.$1.$2")
            .with_key_remap("^_(.+)", "$1");
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

        Ok(record)
    }
}
