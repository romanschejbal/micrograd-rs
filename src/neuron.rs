use crate::value::Value;
use rand::Rng;

#[derive(Debug, Clone)]
pub enum Linearity {
    Linear,
    ReLu,
    Tanh,
}

#[derive(Debug)]
pub struct Neuron<const I: usize> {
    linearity: Linearity,
    weights: [Value; I],
    bias: Value,
}

impl<const I: usize> Neuron<I> {
    pub fn new<R: Rng>(linearity: Linearity, rng: &mut R) -> Self {
        Self {
            linearity,
            weights: core::array::from_fn(|i| {
                Value::new(rng.gen_range(-1.0..1.0), format!("w{i}", i = i + 1))
            }),
            bias: Value::new(rng.gen_range(-1.0..1.0), format!("b")),
        }
    }

    pub fn forward(&self, x: &[Value; I]) -> Value {
        let sum = self
            .weights
            .iter()
            .zip(x)
            .fold(self.bias.clone(), |acc, (w, x)| acc + w.clone() * x.clone());
        match self.linearity {
            Linearity::Linear => sum,
            Linearity::ReLu => sum.relu(),
            Linearity::Tanh => sum.tanh(),
        }
    }

    pub fn nudge(&self, learning_rate: f64) {
        self.bias.nudge(learning_rate);
        for weight in self.weights.iter() {
            weight.nudge(learning_rate);
        }
    }

    pub fn weights(&self) -> &[Value] {
        &self.weights
    }

    pub fn parameters_count(&self) -> usize {
        self.weights.len() + 1 // bias
    }
}
