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

    let xs = (0..100)
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

    let mlp = MultiLayerPerceptron::<1, 1, 3, 1>::new(&mut rng);

    println!("No. of parameters: {}", mlp.parameters().count());

    let mut loss = Value::new(0., "loss");
    let mut total_epochs = 0;

    let model_filename = format!("model_{}.json", mlp.parameters().count());

    // deseserialize if file exists
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

            let lr = 0.01;

            loss.backward();
            mlp.nudge(lr);

            if k % (no_of_epochs.max(10) / 10) == 0 {
                println!(
                    "Iteration: {k: >5} | Loss: {loss: >8.5} | Prediction: {ys:?}",
                    loss = loss.value(),
                    ys = xs
                        .iter()
                        .zip(ys_pred)
                        .take(5)
                        .map(|(_, (y_pred, y_ground))| format!(
                            "{:.2} ~ {:.2}",
                            y_pred[0].value(),
                            y_ground.value()
                        ))
                        .collect::<Vec<_>>()
                        .join(" | ")
                );

                // serialize
                serialize(&model_filename, mlp.parameters(), total_epochs);
            }

            total_epochs += 1;
        }
        serialize(&model_filename, mlp.parameters(), total_epochs);
        let took = now.elapsed().as_secs_f64();
        println!(
            "Final loss: {loss:#?} | Time: {took:.2}s | Total no. of epochs: {total_epochs}",
            loss = loss.value()
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
