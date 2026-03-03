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

    pub fn from_params(linearity: Linearity, params: &[f64]) -> Self {
        assert!(params.len() >= I + 1, "need at least {} params", I + 1);
        Self {
            linearity,
            weights: core::array::from_fn(|i| Value::new(params[i], format!("w{}", i + 1))),
            bias: Value::new(params[I], "b".to_string()),
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

    pub fn zero_grad(&self) {
        self.bias.zero_grad();
        for weight in self.weights.iter() {
            weight.zero_grad();
        }
    }

    pub fn weights(&self) -> &[Value] {
        &self.weights
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.weights.iter().chain(std::iter::once(&self.bias))
    }
}
