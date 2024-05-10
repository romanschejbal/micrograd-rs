# Rust Micrograd with Compile-Time Safety

This project is a Rust-based implementation of the Micrograd engine, originally developed by Andrej Karpathy in Python. This rewrite not only transitions the dynamic capabilities of neural network operations to Rust but also enhances type safety with the use of const generics, ensuring that mismatches between the number of inputs and parameters are caught at compile time.

## Features

- **Compile-Time Safety**: Leveraging Rust's const generics to enforce correct dimensions for inputs and parameters, eliminating a common source of runtime errors.
- **Neural Network Fundamentals**: Implements core neural network structures such as layers and multi-layer perceptrons, with operations like forward and backward propagation.
- **Custom Value and Operation Definitions**: Includes a custom implementation of neural network operations, supporting basic arithmetic, power functions, and non-linear activations like tanh and ReLU.

## Getting Started

### Prerequisites

Ensure you have Rust installed on your machine. You can install Rust through [rustup](https://rustup.rs/).

### Installation

Clone this repository

Run the example neural network:

```bash
cargo run
```

## Usage

This library can be used to define and train neural networks. Here is a basic example of how to use it:

```rust
// Define and train a simple MultiLayer Perceptron
let mut rng = rand::thread_rng();
let mlp = MultiLayerPerceptron::<3, 1, 1, 1>::new(&mut rng);

// Example training loop
for _ in 0..500 {
    let ys_pred = xs.iter().map(|x| mlp.forward(&x)).collect::<Vec<_>>();
    // Mean Squared Error loss calculation
    let loss = ys_pred.iter().enumerate().map(|(i, y_pred)| (y_pred[0].clone() - ys[i].clone()).pow(2.)).sum::<Value>();

    // L2 regularization
    loss = loss + mlp.weights().map(|w| w.clone().pow(2.)).sum::<Value>() * Value::new(0.01, "lambda");

    println!("Loss: {:.4}, Predictions: {:?}", loss.value(), ys_pred.iter().map(|y| y[0].value()).collect::<Vec<_>>()) * Value::new(1. / ys_pred.len() as f64, "n");

    loss.backward();
    mlp.nudge(0.01);
}
```

## Acknowledgments

- Original Micrograd implementation by Andrej Karpathy: [micrograd](https://github.com/karpathy/micrograd)
- This Rust implementation was inspired by the simplicity and educational value of the original Python code.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to discuss potential improvements or fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
