use burn::{
    module::Module,
    tensor::{activation, backend::Backend, Tensor},
};

#[derive(Debug, Clone, Copy, Module, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    Identity,
    Sigmoid,
    Softmax,
    Logsoftmax,
    Tanh,
}

impl Activation {
    pub fn forward<B: Backend>(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Identity => x,
            Self::Sigmoid => activation::sigmoid(x),
            Self::Softmax => activation::softmax(x, 1),
            Self::Logsoftmax => activation::log_softmax(x, 1),
            Self::Tanh => activation::tanh(x),
        }
    }
}
