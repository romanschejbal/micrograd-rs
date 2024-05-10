pub(crate) mod layer;
pub(crate) mod mlp;
pub(crate) mod neuron;
pub(crate) mod value;

use mlp::MultiLayerPerceptron;
use rand::SeedableRng;
use value::Value;

fn main() {
    let xs = vec![
        [
            Value::new(2.0, "x1"),
            Value::new(3.0, "x2"),
            Value::new(-1.0, "x3"),
        ],
        [
            Value::new(3.0, "x1"),
            Value::new(-1., "x2"),
            Value::new(0.5, "x3"),
        ],
        [
            Value::new(0.5, "x1"),
            Value::new(1.0, "x2"),
            Value::new(1.0, "x3"),
        ],
        [
            Value::new(1.0, "x1"),
            Value::new(1.0, "x2"),
            Value::new(-1., "x3"),
        ],
    ];

    let ys = vec![
        Value::new(2., "y1"),
        Value::new(-2., "y2"),
        Value::new(-2., "y3"),
        Value::new(2., "y3"),
    ];

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    // let mut rng = rand::thread_rng();

    let mlp = MultiLayerPerceptron::<3, 1, 1, 1>::new(&mut rng);
    let mut loss = Value::new(0., "loss");

    for k in 0..500000 {
        let ys_pred = xs.iter().map(|x| mlp.forward(&x)).collect::<Vec<_>>();

        // mse loss
        loss = ys_pred
            .iter()
            .enumerate()
            .map(|(i, y_pred)| (y_pred[0].clone() - ys[i].clone()).pow(2.))
            .sum::<Value>();

        // regularization
        loss = loss
            + mlp.weights().map(|w| w.clone().pow(2.)).sum::<Value>() * Value::new(0.01, "lambda");

        if k % 1000 == 0 {
            println!(
                "Iteration: {k: >5} | Loss: {loss: >8.5} | Prediction: {ys:?}",
                loss = loss.value(),
                ys = ys_pred.iter().map(|y| y[0].value()).collect::<Vec<_>>()
            );
        }

        let lr = 0.01;

        loss.backward();
        mlp.nudge(lr);
    }
    println!("Loss: {loss:#?}", loss = loss.value());

    println!("No. of parameters: {}", mlp.parameters_count())
}
