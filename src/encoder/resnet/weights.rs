use super::ResNetConfig;

/// ResNet structure metadata.
pub struct ResNetStructure {
    pub(super) shape: [usize; 4],
    pub(super) url: Option<&'static str>,
    pub(super) num_classes: usize,
}

impl ResNetStructure {
    pub fn to_config(&self) -> ResNetConfig {
        ResNetConfig::new(self.shape, self.num_classes, 1)
    }
}

pub trait WeightsMeta {
    fn weights(&self) -> ResNetStructure;
}

/// ResNet-18 pre-trained weights.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ResNet18 {
    Random {
        num_classes: usize,
    },
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 69.758%.
    /// Top-5 accuracy: 89.078%.
    ImageNet1kV1,
}
impl WeightsMeta for ResNet18 {
    fn weights(&self) -> ResNetStructure {
        let shape = [2, 2, 2, 2];

        let url = match *self {
            ResNet18::Random { num_classes } => {
                return ResNetStructure {
                    shape,
                    url: None,
                    num_classes,
                }
            }
            ResNet18::ImageNet1kV1 => "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        };
        ResNetStructure {
            shape,
            url: Some(url),
            num_classes: 1000,
        }
    }
}

/// ResNet-34 pre-trained weights.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ResNet34 {
    Random {
        num_classes: usize,
    },
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 73.314%.
    /// Top-5 accuracy: 91.420%.
    ImageNet1kV1,
}
impl WeightsMeta for ResNet34 {
    fn weights(&self) -> ResNetStructure {
        let shape = [3, 4, 6, 3];

        let url = match *self {
            ResNet34::Random { num_classes } => {
                return ResNetStructure {
                    shape,
                    url: None,
                    num_classes,
                }
            }
            ResNet34::ImageNet1kV1 => "https://download.pytorch.org/models/resnet34-b627a593.pth",
        };
        ResNetStructure {
            shape,
            url: Some(url),
            num_classes: 1000,
        }
    }
}

/// ResNet-50 pre-trained weights.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ResNet50 {
    Random {
        num_classes: usize,
    },
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 76.130%.
    /// Top-5 accuracy: 92.862%.
    ImageNet1kV1,
    /// These weights improve upon the results of the original paper with a new training
    /// [recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives).
    /// Top-1 accuracy: 80.858%.
    /// Top-5 accuracy: 95.434%.
    ImageNet1kV2,
}
impl WeightsMeta for ResNet50 {
    fn weights(&self) -> ResNetStructure {
        let shape = [3, 4, 6, 3];

        let url = match *self {
            ResNet50::Random { num_classes } => {
                return ResNetStructure {
                    shape,
                    url: None,
                    num_classes,
                }
            }
            ResNet50::ImageNet1kV1 => "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            ResNet50::ImageNet1kV2 => "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        };
        ResNetStructure {
            shape,
            url: Some(url),
            num_classes: 1000,
        }
    }
}

/// ResNet-101 pre-trained weights.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ResNet101 {
    Random {
        num_classes: usize,
    },
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 77.374%.
    /// Top-5 accuracy: 93.546%.
    ImageNet1kV1,
    /// These weights improve upon the results of the original paper with a new training
    /// [recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives).
    /// Top-1 accuracy: 81.886%.
    /// Top-5 accuracy: 95.780%.
    ImageNet1kV2,
}
impl WeightsMeta for ResNet101 {
    fn weights(&self) -> ResNetStructure {
        let shape = [3, 4, 23, 3];

        let url = match *self {
            ResNet101::Random { num_classes } => {
                return ResNetStructure {
                    shape,
                    url: None,
                    num_classes,
                }
            }
            ResNet101::ImageNet1kV1 => "https://download.pytorch.org/models/resnet101-63fe2227.pth",
            ResNet101::ImageNet1kV2 => "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        };
        ResNetStructure {
            shape,
            url: Some(url),
            num_classes: 1000,
        }
    }
}

/// ResNet-152 pre-trained weights.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum ResNet152 {
    Random {
        num_classes: usize,
    },
    /// These weights reproduce closely the results of the original paper.
    /// Top-1 accuracy: 78.312%.
    /// Top-5 accuracy: 94.046%.
    ImageNet1kV1,
    /// These weights improve upon the results of the original paper with a new training
    /// [recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives).
    /// Top-1 accuracy: 82.284%.
    /// Top-5 accuracy: 96.002%.
    ImageNet1kV2,
}
impl WeightsMeta for ResNet152 {
    fn weights(&self) -> ResNetStructure {
        let shape = [3, 8, 36, 3];

        let url = match *self {
            ResNet152::Random { num_classes } => {
                return ResNetStructure {
                    shape,
                    url: None,
                    num_classes,
                }
            }
            ResNet152::ImageNet1kV1 => "https://download.pytorch.org/models/resnet152-394f9c45.pth",
            ResNet152::ImageNet1kV2 => "https://download.pytorch.org/models/resnet152-f82ba261.pth",
        };
        ResNetStructure {
            shape,
            url: Some(url),
            num_classes: 1000,
        }
    }
}
