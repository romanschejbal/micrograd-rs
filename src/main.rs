pub(crate) mod layer;
pub(crate) mod mlp;
pub(crate) mod neuron;
pub(crate) mod value;

use std::io::Write;

use mlp::{compute_sample_gradients, MultiLayerPerceptron};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use value::Value;

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

    let xs: Vec<[f64; 1]> = (0..100)
        .map(|_| {
            let x: f64 = rng.gen_range(-1.0..1.0);
            [x]
        })
        .collect();

    let ys: Vec<[f64; 1]> = xs
        .iter()
        .map(|[x]| [x.powi(2) + x * 2. + 4.])
        .collect();

    let mlp = MultiLayerPerceptron::<1, 1, 3, 1>::new(&mut rng);

    let param_count = mlp.parameters().count();
    println!("No. of parameters: {param_count}");

    let mut total_epochs = 0;

    let model_filename = format!("model_{}.json", param_count);

    // deserialize if file exists
    if let Ok(file) = std::fs::File::open(&model_filename) {
        let (te, values): (usize, Vec<f64>) = serde_json::from_reader(file).unwrap();
        mlp.parameters()
            .zip(values)
            .for_each(|(v, data)| v.set_data(data));
        total_epochs = te;
    }

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

        let now = std::time::Instant::now();
        let mut last_loss = 0.0;

        for k in 0..no_of_epochs {
            let snapshot = mlp.snapshot_params();
            let n_samples = xs.len() as f64;

            // Parallel gradient computation
            let results: Vec<(Vec<f64>, f64)> = xs
                .par_iter()
                .zip(ys.par_iter())
                .map(|(x, y)| compute_sample_gradients::<1, 1, 3, 1>(&snapshot, x, y))
                .collect();

            // Average gradients and loss
            let mut avg_grads = vec![0.0; snapshot.len()];
            let mut avg_loss = 0.0;
            for (grads, loss) in &results {
                avg_loss += loss;
                for (ag, g) in avg_grads.iter_mut().zip(grads.iter()) {
                    *ag += g;
                }
            }
            avg_loss /= n_samples;
            for ag in avg_grads.iter_mut() {
                *ag /= n_samples;
            }

            last_loss = avg_loss;

            let lr = 0.01;
            mlp.zero_grad();
            mlp.accumulate_gradients(&avg_grads);
            mlp.nudge(lr);

            if k % (no_of_epochs.max(10) / 10) == 0 {
                // Quick forward pass for display predictions
                let predictions: Vec<String> = xs
                    .iter()
                    .zip(ys.iter())
                    .take(5)
                    .map(|(x, y)| {
                        let x_val = [Value::new(x[0], "x")];
                        let y_pred = mlp.forward(&x_val);
                        format!("{:.2} ~ {:.2}", y_pred[0].value(), y[0])
                    })
                    .collect();

                println!(
                    "Iteration: {k: >5} | Loss: {avg_loss: >8.5} | Prediction: {preds:?}",
                    preds = predictions.join(" | ")
                );

                // serialize
                serialize(&model_filename, mlp.parameters(), total_epochs);
            }

            total_epochs += 1;
        }
        serialize(&model_filename, mlp.parameters(), total_epochs);
        let took = now.elapsed().as_secs_f64();
        println!(
            "Final loss: {last_loss:#?} | Time: {took:.2}s | Total no. of epochs: {total_epochs}",
        );
    }
}

fn serialize<'a>(
    filename: &'a str,
    parameters: impl Iterator<Item = &'a Value>,
    total_epochs: usize,
) {
    let values = parameters.map(|v| v.data()).collect::<Vec<_>>();
    serde_json::to_writer_pretty(
        std::fs::File::create(&filename).unwrap(),
        &(total_epochs, values),
    )
    .unwrap();
}
