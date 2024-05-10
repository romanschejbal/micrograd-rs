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

    pub fn forward(&self, x: &[Value; I]) -> [Value; O] {
        core::array::from_fn(|i| self.neurons[i].forward(x))
    }

    pub fn nudge(&self, learning_rate: f64) {
        for neuron in self.neurons.iter() {
            neuron.nudge(learning_rate);
        }
    }

    pub fn weights(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|n| n.weights())
    }

    pub fn parameters_count(&self) -> usize {
        self.neurons.iter().map(|n| n.parameters_count()).sum()
    }
}
