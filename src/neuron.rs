use crate::value::Value;
use rand::Rng;

#[derive(Debug, Clone)]
pub enum Linearity {
    Linear,
    ReLu,
    Tanh,
    Sigmoid,
}

#[derive(Debug)]
pub struct InputNeuron<const I: usize> {
    inner: Neuron,
}

impl<const I: usize> InputNeuron<I> {
    pub fn new<R: Rng>(linearity: Linearity, rng: &mut R) -> Self {
        Self {
            inner: Neuron::new(linearity, I, rng),
        }
    }
}

impl<const I: usize> AsRef<Neuron> for InputNeuron<I> {
    fn as_ref(&self) -> &Neuron {
        &self.inner
    }
}

#[derive(Debug)]
pub struct Neuron {
    linearity: Linearity,
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new<R: Rng>(linearity: Linearity, weights: usize, rng: &mut R) -> Self {
        Self {
            linearity,
            weights: (0..weights)
                .map(|i| Value::new(rng.gen_range(-1.0..1.0), format!("w{i}", i = i + 1)))
                .collect(),
            bias: Value::new(rng.gen_range(-1.0..1.0), format!("b")),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        let sum = self
            .weights
            .iter()
            .zip(x)
            .fold(self.bias.clone(), |acc, (w, x)| acc + w.clone() * x.clone());
        match self.linearity {
            Linearity::Linear => sum,
            Linearity::ReLu => sum.relu(),
            Linearity::Tanh => sum.tanh(),
            Linearity::Sigmoid => sum.sigmoid(),
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

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.weights.iter().chain(std::iter::once(&self.bias))
    }
}

impl AsRef<Neuron> for Neuron {
    fn as_ref(&self) -> &Neuron {
        self
    }
}
