use crate::{layer::Layer, value::Value};
use rand::Rng;

#[derive(Debug)]
pub struct MultiLayerPerceptron<const I: usize, const H: usize, const N: usize, const O: usize> {
    input_layer: Layer<I, N>,
    hidden_layers: [Layer<N, N>; H],
    output_layer: Layer<N, O>,
}

impl<const I: usize, const H: usize, const N: usize, const O: usize>
    MultiLayerPerceptron<I, H, N, O>
{
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        Self {
            input_layer: Layer::new(true, rng),
            hidden_layers: core::array::from_fn(|_| Layer::new(true, rng)),
            output_layer: Layer::new(false, rng),
        }
    }

    pub fn forward(&self, x: &[Value; I]) -> [Value; O] {
        let mut x = self.input_layer.forward(x);
        for layer in self.hidden_layers.iter() {
            x = layer.forward(&x);
        }
        self.output_layer.forward(&x)
    }

    pub fn nudge(&self, learning_rate: f64) {
        self.input_layer.nudge(learning_rate);
        for layer in self.hidden_layers.iter() {
            layer.nudge(learning_rate);
        }
        self.output_layer.nudge(learning_rate);
    }

    pub fn parameters_count(&self) -> usize {
        self.input_layer.parameters_count()
            + self
                .hidden_layers
                .iter()
                .map(|l| l.parameters_count())
                .sum::<usize>()
            + self.output_layer.parameters_count()
    }
}
