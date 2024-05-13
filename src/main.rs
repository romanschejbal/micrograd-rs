pub(crate) mod layer;
// pub(crate) mod lstm;
pub(crate) mod mlp;
pub(crate) mod neuron;
// pub(crate) mod rnn;
pub(crate) mod value;

// use crate::{lstm::LSTM, rnn::RecurrentNeuralNetwork};
use clap::Parser;
use mlp::MultiLayerPerceptron;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::io::Write;
use value::Value;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[clap(short, long, default_value = "1")]
    hidden_layers: usize,

    #[clap(short, long, default_value = "1")]
    neurons_per_layer: usize,
}

fn main() {
    let args = Cli::parse();

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

    // // LSTM
    // let mut lstm = LSTM::<1, 1>::new(&mut rng);
    // loop {
    //     let mut buf = String::new();
    //     print!("Enter value: ");
    //     std::io::stdout().flush().unwrap();
    //     std::io::stdin().read_line(&mut buf).unwrap();
    //     let Ok(value) = buf.trim().parse::<f64>() else {
    //         continue;
    //     };

    //     let next = lstm.step(&[Value::new(value, "x")]);
    //     let loss = (next[0].clone() - Value::new(value, "y")).pow(2.);

    //     loss.backward();
    //     lstm.nudge(0.01);

    //     println!("Prediction: {} | Loss: {}", next[0].value(), loss.value());
    // }

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

    let mut mlp = MultiLayerPerceptron::<1>::new(
        &mut rng,
        args.hidden_layers,
        args.neurons_per_layer,
        neuron::Linearity::Tanh,
    );

    println!("No. of parameters: {}", mlp.parameters().count());

    let mut loss = Value::new(0., "loss");
    let mut total_epochs = 0;

    let model_filename = format!("model_{}.json", mlp.parameters().count());

    // deseserialize if file exists
    if let Ok(file) = std::fs::File::open(&model_filename) {
        let model: Model = serde_json::from_reader(file).unwrap();
        mlp.parameters()
            .zip(model.parameters)
            .for_each(|(v, data)| v.set_data(data));
        total_epochs = model.total_epochs;
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

        let mut loss_final = 0.;
        let now = std::time::Instant::now();
        for k in 0..no_of_epochs {
            let ys_pred = xs
                .iter()
                .zip(ys_ground.iter())
                .map(|(x, y)| (mlp.forward(x).to_vec(), y))
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
                serialize(
                    &model_filename,
                    mlp.parameters(),
                    total_epochs,
                    loss.value(),
                );
                loss_final = loss.value();
            }

            total_epochs += 1;
        }
        serialize(&model_filename, mlp.parameters(), total_epochs, loss_final);
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
    loss: f64,
) {
    let parameters = parameters.map(|v| v.data()).collect::<Vec<_>>();
    serde_json::to_writer_pretty(
        std::fs::File::create(&filename).unwrap(),
        &Model {
            total_epochs,
            parameters,
            loss,
        },
    )
    .unwrap();
}

#[derive(Serialize, Deserialize)]
struct Model {
    loss: f64,
    total_epochs: usize,
    parameters: Vec<f64>,
}
