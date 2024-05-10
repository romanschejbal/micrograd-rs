pub(crate) mod layer;
pub(crate) mod mlp;
pub(crate) mod neuron;
pub(crate) mod value;

use std::io::Write;

use mlp::MultiLayerPerceptron;
use rand::{Rng, SeedableRng};
use value::Value;

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

    let xs = (0..500)
        .into_iter()
        .map(|_| {
            let x = rng.gen_range(-1.0..1.0);
            [x]
        })
        .collect::<Vec<_>>();

    let ys_ground = xs
        .iter()
        .map(|[x]| Value::new((*x as f64).powi(2) + x * 2. + 4., "y"))
        .collect::<Vec<_>>();

    let xs = xs
        .into_iter()
        .map(|[x]| [Value::new(x, "x")]) //, Value::new(a, "a"), Value::new(b, "b")])
        .collect::<Vec<_>>();

    let mlp = MultiLayerPerceptron::<1, 1, 10, 1>::new(&mut rng);
    println!("No. of parameters: {}", mlp.parameters_count());

    let mut loss = Value::new(0., "loss");
    let mut total_epochs = 0;

    loop {
        let mut buf = String::new();
        print!("Enter no. of epochs: ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut buf).unwrap();
        let Ok(no_of_epochs) = buf.trim().parse::<usize>() else {
            continue;
        };
        if no_of_epochs == 0 {
            break;
        }

        for k in 0..no_of_epochs {
            let ys_pred = xs
                .iter()
                .zip(ys_ground.iter())
                .map(|(x, y)| (mlp.forward(&x), y))
                .collect::<Vec<_>>();

            // mse loss
            loss = ys_pred
                .iter()
                .map(|(y_pred, y_ground)| (y_pred[0].clone() - (*y_ground).clone()).pow(2.))
                .sum::<Value>()
                * Value::new(1. / ys_pred.len() as f64, "n");

            // regularization
            loss = loss
                + mlp.weights().map(|w| w.clone().pow(2.)).sum::<Value>()
                    * Value::new(0.01, "lambda");

            if k % (no_of_epochs.max(10) / 10) == 0 {
                println!(
                    "Iteration: {k: >5} | Loss: {loss: >8.5} | Prediction: {ys:?}",
                    loss = loss.value(),
                    ys = xs
                        .iter()
                        .zip(ys_pred)
                        .skip(3)
                        .take(3)
                        .map(|(_, (y_pred, y_ground))| format!(
                            "{:.2} ~ {:.2}",
                            y_pred[0].value(),
                            y_ground.value()
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            let lr = 0.01;

            loss.backward();
            mlp.nudge(lr);
            total_epochs += 1;
        }
        println!(
            "Final loss: {loss:#?} | Total no. of epochs: {total_epochs}",
            loss = loss.value()
        );
    }
}
