use crate::{layer::Layer, neuron::Linearity, value::Value};
use rand::Rng;

#[derive(Debug)]
pub struct MultiLayerPerceptron<const I: usize, const H: usize, const N: usize, const O: usize> {
    input_layer: Layer<I, N>,
    hidden_layers: [Layer<N, N>; H],
    output_layer: Layer<N, O>,
}

impl<const I: usize, const H: usize, const N: usize, const O: usize>
    MultiLayerPerceptron<I, H, N, O>
{
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        Self {
            input_layer: Layer::new(Linearity::Tanh, rng),
            hidden_layers: core::array::from_fn(|_| Layer::new(Linearity::ReLu, rng)),
            output_layer: Layer::new(Linearity::Linear, rng),
        }
    }

    pub fn from_params(params: &[f64]) -> Self {
        let input_layer_size = N * (I + 1);
        let hidden_layer_size = N * (N + 1);
        let output_layer_size = O * (N + 1);
        let total = input_layer_size + H * hidden_layer_size + output_layer_size;
        assert!(
            params.len() >= total,
            "need at least {} params, got {}",
            total,
            params.len()
        );

        let mut offset = 0;
        let input_layer = Layer::from_params(Linearity::Tanh, &params[offset..]);
        offset += input_layer_size;

        let hidden_layers = core::array::from_fn(|_| {
            let layer = Layer::from_params(Linearity::ReLu, &params[offset..]);
            offset += hidden_layer_size;
            layer
        });

        let output_layer = Layer::from_params(Linearity::Linear, &params[offset..]);

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }

    pub fn forward(&self, x: &[Value; I]) -> [Value; O] {
        let mut x = self.input_layer.forward(x);
        for layer in self.hidden_layers.iter() {
            x = layer.forward(&x);
        }
        self.output_layer.forward(&x)
    }

    pub fn nudge(&self, learning_rate: f64) {
        self.input_layer.nudge(learning_rate);
        for layer in self.hidden_layers.iter() {
            layer.nudge(learning_rate);
        }
        self.output_layer.nudge(learning_rate);
    }

    pub fn zero_grad(&self) {
        self.input_layer.zero_grad();
        for layer in self.hidden_layers.iter() {
            layer.zero_grad();
        }
        self.output_layer.zero_grad();
    }

    pub fn snapshot_params(&self) -> Vec<f64> {
        self.parameters().map(|v| v.data()).collect()
    }

    pub fn accumulate_gradients(&self, grads: &[f64]) {
        for (param, grad) in self.parameters().zip(grads.iter()) {
            param.add_gradient(*grad);
        }
    }

    pub fn weights(&self) -> impl Iterator<Item = &Value> {
        self.input_layer.weights().chain(
            self.hidden_layers
                .iter()
                .flat_map(|l| l.weights())
                .chain(self.output_layer.weights()),
        )
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.input_layer.parameters().chain(
            self.hidden_layers
                .iter()
                .flat_map(|l| l.parameters())
                .chain(self.output_layer.parameters()),
        )
    }
}

pub fn compute_sample_gradients<const I: usize, const H: usize, const N: usize, const O: usize>(
    params: &[f64],
    input: &[f64; I],
    target: &[f64; O],
) -> (Vec<f64>, f64) {
    let mlp = MultiLayerPerceptron::<I, H, N, O>::from_params(params);

    let x: [Value; I] = core::array::from_fn(|i| Value::new(input[i], "x"));
    let y_pred = mlp.forward(&x);

    // Squared error loss
    let mut loss: Value = core::array::from_fn::<_, O, _>(|i| {
        (y_pred[i].clone() - Value::new(target[i], "y")).pow(2.)
    })
    .into_iter()
    .sum::<Value>();

    // L2 regularization
    loss = loss
        + mlp.weights().map(|w| w.clone().pow(2.)).sum::<Value>()
            * Value::new(0.01, "lambda");

    loss.backward();

    let grads: Vec<f64> = mlp.parameters().map(|v| v.gradient()).collect();
    (grads, loss.value())
}
