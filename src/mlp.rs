use crate::{
    layer::{InputLayer, Layer},
    neuron::{Linearity, Neuron},
    value::Value,
};
use rand::Rng;

#[derive(Debug)]
pub struct MultiLayerPerceptron<const I: usize> {
    input_layer: InputLayer<I>,
    hidden_layers: Vec<Layer<Neuron>>,
    output_layer: Layer<Neuron>,
}

impl<const I: usize> MultiLayerPerceptron<I> {
    pub fn new<R: Rng>(
        rng: &mut R,
        hidden_layers: usize,
        neurons: usize,
        linearity: Linearity,
    ) -> Self {
        Self {
            input_layer: InputLayer::new(linearity.clone(), neurons, rng),
            hidden_layers: (0..hidden_layers)
                .map(|_| Layer::<Neuron>::new(linearity.clone(), neurons, neurons, rng))
                .collect(),
            output_layer: Layer::<Neuron>::new(Linearity::Linear, 1, neurons, rng),
        }
    }

    pub fn forward(&mut self, x: &[Value; I]) -> &[Value] {
        let mut x = self.input_layer.as_mut().forward(x);
        for layer in self.hidden_layers.iter_mut() {
            x = layer.forward(&x);
        }
        self.output_layer.forward(&x)
    }

    pub fn nudge(&self, learning_rate: f64) {
        self.input_layer.as_ref().nudge(learning_rate);
        for layer in self.hidden_layers.iter() {
            layer.nudge(learning_rate);
        }
        self.output_layer.nudge(learning_rate);
    }

    pub fn weights(&self) -> impl Iterator<Item = &Value> {
        self.input_layer.as_ref().weights().chain(
            self.hidden_layers
                .iter()
                .flat_map(|l| l.weights())
                .chain(self.output_layer.weights()),
        )
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.input_layer.as_ref().parameters().chain(
            self.hidden_layers
                .iter()
                .flat_map(|l| l.parameters())
                .chain(self.output_layer.parameters()),
        )
    }
}
