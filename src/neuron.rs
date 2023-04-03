use crate::value::Value;
use rand::Rng;

#[derive(Debug)]
pub struct Neuron<const I: usize> {
    nonlin: bool,
    weights: [Value; I],
    bias: Value,
}

impl<const I: usize> Neuron<I> {
    pub fn new<R: Rng>(nonlin: bool, rng: &mut R) -> Self {
        Self {
            nonlin,
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
        if self.nonlin {
            sum.tanh()
        } else {
            sum
        }
    }

    pub fn nudge(&self, learning_rate: f64) {
        self.bias.nudge(learning_rate);
        for weight in self.weights.iter() {
            weight.nudge(learning_rate);
        }
    }

    pub fn parameters_count(&self) -> usize {
        self.weights.len() + 1 // bias
    }
}
