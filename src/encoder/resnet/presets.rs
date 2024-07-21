use burn::{
    config::Config,
    record::RecorderError,
    tensor::{backend::Backend, Device},
};

use crate::encoder::EncoderConfig;

use super::{weights::*, ResNet};

#[derive(Debug, Config)]
pub enum ResNetConfigPreset {
    ResNet18(ResNet18),
    ResNet34(ResNet34),
    ResNet50(ResNet50),
    ResNet101(ResNet101),
    ResNet152(ResNet152),
}

impl ResNetConfigPreset {
    pub fn resnet_pretrained<B: Backend>(
        &self,
        device: &Device<B>,
    ) -> Result<ResNet<B>, RecorderError> {
        let structure = self.structure();
        let model = structure.to_config().init(device);

        match structure.url {
            #[cfg(feature = "pretrained")]
            Some(url) => {
                use burn::module::Module;

                let weight = crate::download("resnet", url).map_err(|err| {
                    RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
                })?;
                let record = self.load_weights_record(weight, device)?;
                Ok(model.load_record(record))
            }
            None => Ok(model),
            #[allow(unreachable_patterns)]
            _ => panic!("Please enable `pretrained` feature to use pretrained weights."),
        }
    }

    fn structure(&self) -> ResNetStructure {
        match self {
            Self::ResNet18(resnet18) => resnet18.weights(),
            Self::ResNet34(resnet34) => resnet34.weights(),
            Self::ResNet50(resnet50) => resnet50.weights(),
            Self::ResNet101(resnet101) => resnet101.weights(),
            Self::ResNet152(resnet152) => resnet152.weights(),
        }
    }

    /// Load specified pre-trained PyTorch weights as a record.
    #[cfg(feature = "pretrained")]
    fn load_weights_record<B: Backend, P: AsRef<std::path::Path>>(
        &self,
        torch_weights: P,
        device: &Device<B>,
    ) -> Result<super::ResNetRecord<B>, RecorderError> {
        use burn::record::{FullPrecisionSettings, Recorder};
        use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

        // Load weights from torch state_dict
        let load_args = LoadArgs::new(torch_weights.as_ref().into())
            // Map *.downsample.0.* -> *.downsample.conv.*
            .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
            // Map *.downsample.1.* -> *.downsample.bn.*
            .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
            // Map layer[i].[j].* -> layer[i].blocks.[j].*
            .with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");
        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

        Ok(record)
    }
}

impl<B: Backend> EncoderConfig<B> for ResNetConfigPreset {
    type Encoder = ResNet<B>;
    fn init(&self, device: &B::Device) -> Self::Encoder {
        self.resnet_pretrained(device).unwrap()
    }
    fn out_channels(&self) -> Vec<usize> {
        match self {
            Self::ResNet18(_) => vec![3, 64, 64, 128, 256, 512],
            Self::ResNet34(_) => vec![3, 64, 64, 128, 256, 512],
            Self::ResNet50(_) => vec![3, 64, 256, 512, 1024, 2048],
            Self::ResNet101(_) => vec![3, 64, 256, 512, 1024, 2048],
            Self::ResNet152(_) => vec![3, 64, 256, 512, 1024, 2048],
        }
    }
    fn feature_idxs(&self) -> Vec<usize> {
        vec![]
    }
}
