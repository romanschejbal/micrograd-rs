use crate::{
    neuron::{InputNeuron, Linearity, Neuron},
    value::Value,
};
use rand::Rng;

#[derive(Debug)]
pub struct Layer<N>
where
    N: AsRef<Neuron>,
{
    neurons: Vec<N>,
    // to avoid reallocation on each forward pass
    result: Vec<Value>,
}

impl<N> Layer<N>
where
    N: AsRef<Neuron>,
{
    pub fn forward(&mut self, x: &[Value]) -> &[Value] {
        for (i, neuron) in self.neurons.iter().enumerate() {
            self.result[i] = neuron.as_ref().forward(x);
        }
        &self.result
    }

    pub fn nudge(&self, learning_rate: f64) {
        for neuron in self.neurons.iter() {
            neuron.as_ref().nudge(learning_rate);
        }
    }

    pub fn weights(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|n| n.as_ref().weights())
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|n| n.as_ref().parameters())
    }
}

impl Layer<Neuron> {
    pub fn new<R: Rng>(linearity: Linearity, neurons: usize, weights: usize, rng: &mut R) -> Self {
        Self {
            neurons: (0..neurons)
                .map(|_| Neuron::new(linearity.clone(), weights, rng))
                .collect::<Vec<_>>(),
            result: (0..neurons)
                .map(|i| Value::new(0., format!("y{i}")))
                .collect::<Vec<_>>(),
        }
    }
}

impl<const I: usize> Layer<InputNeuron<I>> {
    pub fn new<R: Rng>(linearity: Linearity, neurons: usize, rng: &mut R) -> Self {
        Self {
            neurons: (0..neurons)
                .map(|_| InputNeuron::new(linearity.clone(), rng))
                .collect::<Vec<_>>(),
            result: (0..neurons)
                .map(|i| Value::new(0., format!("y{i}")))
                .collect::<Vec<_>>(),
        }
    }
}

#[derive(Debug)]
pub struct InputLayer<const I: usize> {
    inner: Layer<InputNeuron<I>>,
}

impl<const I: usize> InputLayer<I> {
    pub fn new<R: Rng>(linearity: Linearity, neurons: usize, rng: &mut R) -> Self {
        Self {
            inner: Layer::<InputNeuron<I>>::new(linearity, neurons, rng),
        }
    }
}

impl<const I: usize> AsRef<Layer<InputNeuron<I>>> for InputLayer<I> {
    fn as_ref(&self) -> &Layer<InputNeuron<I>> {
        &self.inner
    }
}

impl<const I: usize> AsMut<Layer<InputNeuron<I>>> for InputLayer<I> {
    fn as_mut(&mut self) -> &mut Layer<InputNeuron<I>> {
        &mut self.inner
    }
}
