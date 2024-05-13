use rand::Rng;

use crate::{layer::Layer, neuron::Linearity, rnn::RecurrentNeuralNetwork, value::Value};

#[derive(Debug)]
pub struct LSTM<const I: usize, const N: usize> {
    rnn: RecurrentNeuralNetwork<I, N>,
    forget_layer: Layer<N, N>,
    output_layer: Layer<N, N>,
    cell_state: [Value; N],
}

impl<const I: usize, const N: usize> LSTM<I, N> {
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        Self {
            rnn: RecurrentNeuralNetwork::new(rng),
            forget_layer: Layer::new(Linearity::Sigmoid, rng),
            output_layer: Layer::new(Linearity::Sigmoid, rng),
            cell_state: core::array::from_fn(|_| Value::new(0., "cell_state")),
        }
    }

    pub fn step(&mut self, x: &[Value; I]) -> [Value; N] {
        let hidden_output = self.rnn.step(x);
        let forget_gate = self.forget_layer.forward(&hidden_output);
        let output_gate = self.output_layer.forward(&hidden_output);

        self.cell_state = self
            .cell_state
            .iter()
            .zip(forget_gate.iter())
            .map(|(c, f)| c.clone() * f.clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| self.cell_state.clone());

        self.cell_state = self
            .cell_state
            .iter()
            .zip(hidden_output.iter())
            .map(|(c, h)| c.clone() + h.clone()) // Cell state update (could be refined)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| self.cell_state.clone());

        let final_output = self
            .cell_state
            .iter()
            .zip(output_gate.iter())
            .map(|(c, o)| c.clone() * o.clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or_else(|_| self.cell_state.clone());

        final_output
    }

    pub fn nudge(&self, learning_rate: f32) {
        self.rnn.nudge(learning_rate);
        self.forget_layer.nudge(learning_rate);
        self.output_layer.nudge(learning_rate);
        self.cell_state.iter().for_each(|v| v.nudge(learning_rate));
    }
}
