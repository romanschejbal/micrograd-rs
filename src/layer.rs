use crate::{
    neuron::{Linearity, Neuron},
    value::Value,
};
use rand::Rng;

#[derive(Debug)]
pub struct Layer<const I: usize, const O: usize> {
    neurons: [Neuron<I>; O],
}

impl<const I: usize, const O: usize> Layer<I, O> {
    pub fn new<R: Rng>(linearity: Linearity, rng: &mut R) -> Self {
        Self {
            neurons: core::array::from_fn(|_| Neuron::new(linearity.clone(), rng)),
        }
    }

    pub fn from_params(linearity: Linearity, params: &[f64]) -> Self {
        let params_per_neuron = I + 1;
        assert!(
            params.len() >= O * params_per_neuron,
            "need at least {} params",
            O * params_per_neuron
        );
        Self {
            neurons: core::array::from_fn(|i| {
                let offset = i * params_per_neuron;
                Neuron::from_params(linearity.clone(), &params[offset..offset + params_per_neuron])
            }),
        }
    }

    pub fn forward(&self, x: &[Value; I]) -> [Value; O] {
        core::array::from_fn(|i| self.neurons[i].forward(x))
    }

    pub fn nudge(&self, learning_rate: f64) {
        for neuron in self.neurons.iter() {
            neuron.nudge(learning_rate);
        }
    }

    pub fn zero_grad(&self) {
        for neuron in self.neurons.iter() {
            neuron.zero_grad();
        }
    }

    pub fn weights(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|n| n.weights())
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|n| n.parameters())
    }
}
