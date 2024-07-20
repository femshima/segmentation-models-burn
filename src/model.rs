use burn::{
    module::{AutodiffModule, Devices, ModuleMapper, ModuleVisitor},
    prelude::*,
    record::Record,
    tensor::backend::AutodiffBackend,
};

use crate::{
    decoder::{Decoder, DecoderConfig},
    encoder::{Encoder, EncoderConfig},
    segmentation_head::{SegmentationHead, SegmentationHeadConfig},
};

#[derive(Debug, Clone)]
pub struct Model<B: Backend, E, D> {
    pub encoder: E,
    pub decoder: D,
    pub head: SegmentationHead<B>,
    pub feature_idxs: Vec<usize>,
}

impl<B: Backend, E: Encoder<Backend = B>, D: Decoder<Backend = B>> Model<B, E, D> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let features = self.encoder.forward(x, self.feature_idxs.clone());
        let x = self.decoder.forward(features);
        self.head.forward(x)
    }
}

// We need to implement everything manually, because [`Module`] proc-macro implementation has unnecessary trait bound for AutodiffModule.
#[derive(Record)]
pub struct ModelRecord<B: Backend, E, D>
where
    E: burn::module::Module<B>,
    D: burn::module::Module<B>,
{
    pub encoder: <E as burn::module::Module<B>>::Record,
    pub decoder: <D as burn::module::Module<B>>::Record,
    pub head: <SegmentationHead<B> as burn::module::Module<B>>::Record,
    pub feature_idxs: <Vec<usize> as burn::module::Module<B>>::Record,
}

impl<B: Backend, E: Module<B>, D: Module<B>> Module<B> for Model<B, E, D> {
    type Record = ModelRecord<B, E, D>;
    fn load_record(self, record: Self::Record) -> Self {
        Self {
            encoder: Module::<B>::load_record(self.encoder, record.encoder),
            decoder: Module::<B>::load_record(self.decoder, record.decoder),
            head: Module::<B>::load_record(self.head, record.head),
            feature_idxs: Module::<B>::load_record(self.feature_idxs, record.feature_idxs),
        }
    }
    fn into_record(self) -> Self::Record {
        Self::Record {
            encoder: Module::<B>::into_record(self.encoder),
            decoder: Module::<B>::into_record(self.decoder),
            head: Module::<B>::into_record(self.head),
            feature_idxs: Module::<B>::into_record(self.feature_idxs),
        }
    }
    fn num_params(&self) -> usize {
        Module::<B>::num_params(&self.encoder)
            + Module::<B>::num_params(&self.decoder)
            + Module::<B>::num_params(&self.head)
            + Module::<B>::num_params(&self.feature_idxs)
    }
    fn visit<Visitor: ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        Module::visit(&self.encoder, visitor);
        Module::visit(&self.decoder, visitor);
        Module::visit(&self.head, visitor);
        Module::visit(&self.feature_idxs, visitor);
    }
    fn map<Mapper: ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        let encoder = Module::<B>::map(self.encoder, mapper);
        let decoder = Module::<B>::map(self.decoder, mapper);
        let head = Module::<B>::map(self.head, mapper);
        let feature_idxs = Module::<B>::map(self.feature_idxs, mapper);
        Self {
            encoder,
            decoder,
            head,
            feature_idxs,
        }
    }
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        let devices = Module::<B>::collect_devices(&self.encoder, devices);
        let devices = Module::<B>::collect_devices(&self.decoder, devices);
        let devices = Module::<B>::collect_devices(&self.head, devices);
        let devices = Module::<B>::collect_devices(&self.feature_idxs, devices);
        devices
    }
    fn to_device(self, device: &B::Device) -> Self {
        let encoder = Module::<B>::to_device(self.encoder, device);
        let decoder = Module::<B>::to_device(self.decoder, device);
        let head = Module::<B>::to_device(self.head, device);
        let feature_idxs = Module::<B>::to_device(self.feature_idxs, device);
        Self {
            encoder,
            decoder,
            head,
            feature_idxs,
        }
    }
    fn fork(self, device: &B::Device) -> Self {
        let encoder = Module::<B>::fork(self.encoder, device);
        let decoder = Module::<B>::fork(self.decoder, device);
        let head = Module::<B>::fork(self.head, device);
        let feature_idxs = Module::<B>::fork(self.feature_idxs, device);
        Self {
            encoder,
            decoder,
            head,
            feature_idxs,
        }
    }
}

impl<B: AutodiffBackend, E: AutodiffModule<B>, D: AutodiffModule<B>> AutodiffModule<B>
    for Model<B, E, D>
{
    type InnerModule = Model<
        B::InnerBackend,
        <E as AutodiffModule<B>>::InnerModule,
        <D as AutodiffModule<B>>::InnerModule,
    >;
    fn valid(&self) -> Self::InnerModule {
        let encoder = AutodiffModule::<B>::valid(&self.encoder);
        let decoder = AutodiffModule::<B>::valid(&self.decoder);
        let head = AutodiffModule::<B>::valid(&self.head);
        let feature_idxs = AutodiffModule::<B>::valid(&self.feature_idxs);
        Self::InnerModule {
            encoder,
            decoder,
            head,
            feature_idxs,
        }
    }
}

impl<B: Backend, E: Module<B>, D: Module<B>> core::fmt::Display for Model<B, E, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "{0}[num_params={1}]",
            "Model",
            self.num_params()
        ))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig<EC, DC> {
    encoder: EC,
    decoder: DC,
    feature_idxs: Vec<usize>,
}

impl<EC, DC> ModelConfig<EC, DC> {
    pub fn new<B: Backend>(encoder: EC, decoder: DC) -> ModelConfig<EC, DC>
    where
        EC: EncoderConfig<B>,
        DC: DecoderConfig<B>,
    {
        let decoder = decoder.with_encoder_channels(encoder.out_channels());
        let feature_idxs = encoder.feature_idxs();
        ModelConfig {
            encoder,
            decoder,
            feature_idxs,
        }
    }
}

pub trait ModelInit<B: Backend, EC: EncoderConfig<B>, DC: DecoderConfig<B>> {
    fn init(&self, classes: usize, device: &B::Device) -> Model<B, EC::Encoder, DC::Decoder>;
}

impl<B: Backend, EC: EncoderConfig<B>, DC: DecoderConfig<B>> ModelInit<B, EC, DC>
    for ModelConfig<EC, DC>
{
    fn init(&self, classes: usize, device: &B::Device) -> Model<B, EC::Encoder, DC::Decoder> {
        let encoder = self.encoder.init(device);
        let decoder = self
            .decoder
            .with_encoder_channels(self.encoder.out_channels())
            .init(device);
        let head = SegmentationHeadConfig::new(self.decoder.out_channels(), classes).init(device);
        Model {
            encoder,
            decoder,
            head,
            feature_idxs: self.feature_idxs.clone(),
        }
    }
}
