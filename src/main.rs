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

    for _k in 0..50 {
        let ys_pred = xs.iter().map(|x| mlp.forward(&x)).collect::<Vec<_>>();

        // mse loss
        let loss = ys_pred
            .iter()
            .enumerate()
            .map(|(i, y_pred)| (y_pred[0].clone() - ys[i].clone()).pow(2.))
            .sum::<Value>();

        let lr = 0.01;

        println!(
            "Loss: {loss:#?}, Predictions: {ys:?}, LR = {lr}",
            loss = loss.value(),
            ys = ys_pred.iter().map(|y| y[0].value()).collect::<Vec<_>>()
        );

        loss.backward();
        mlp.nudge(lr);
    }

    println!("No. of parameters: {}", mlp.parameters_count())
}
