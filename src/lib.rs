#[cfg(feature = "pretrained")]
mod download;
#[cfg(feature = "pretrained")]
pub(crate) use download::download;

pub mod decoder;
pub mod encoder;
pub mod segmentation_head;

mod model;
pub use model::*;
