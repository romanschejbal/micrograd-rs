use crate::{layer::Layer, neuron::Linearity, value::Value};
use rand::Rng;

#[derive(Debug)]
pub struct RecurrentNeuralNetwork<const I: usize, const N: usize> {
    input_layer: Layer<I, N>,
    hidden_layer: Layer<N, N>,
    hidden_state: [Value; N],
}

impl<const I: usize, const N: usize> RecurrentNeuralNetwork<I, N> {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        Self {
            input_layer: Layer::new(Linearity::Tanh, rng),
            hidden_layer: Layer::new(Linearity::Tanh, rng),
            hidden_state: core::array::from_fn(|_| Value::new(0., "hidden_state")),
        }
    }

    pub fn step(&mut self, x: &[Value; I]) -> [Value; N] {
        let input_activation = self.input_layer.forward(x);
        let combined_input = input_activation
            .iter()
            .zip(self.hidden_state.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| self.hidden_state.clone());

        self.hidden_state = self.hidden_layer.forward(&combined_input);
        self.hidden_state.clone()
    }

    pub fn nudge(&self, learning_rate: f32) {
        self.input_layer.nudge(learning_rate);
        self.hidden_layer.nudge(learning_rate);
        self.hidden_state
            .iter()
            .for_each(|v| v.nudge(learning_rate));
    }
}
