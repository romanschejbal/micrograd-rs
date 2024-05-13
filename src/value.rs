use std::{
    cell::RefCell,
    iter::Sum,
    ops::{Add, Mul, Sub},
    rc::Rc,
};

#[derive(Debug, Clone)]
pub struct Value {
    inner: SharedValueInner,
}

impl Value {
    pub fn new(data: f64, label: impl Into<String>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                data,
                op: Operation::Const,
                label: label.into(),
                gradient: 0.,
            })),
        }
    }

    fn new_with_children(data: f64, label: impl Into<String>, op: Operation) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                data,
                label: label.into(),
                op,
                gradient: 0.,
            })),
        }
    }

    pub fn pow(self, exp: f64) -> Self {
        Self::new_with_children(
            self.inner.borrow().data.powf(exp),
            format!("{}^{exp}", self.inner.borrow().label),
            Operation::Pow(self.inner.clone(), exp),
        )
    }

    pub fn tanh(self) -> Self {
        Self::new_with_children(
            self.inner.borrow().data.tanh(),
            format!("tanh({})", self.inner.borrow().label),
            Operation::Tanh(self.inner.clone()),
        )
    }

    pub fn sigmoid(self) -> Self {
        let data = self.inner.borrow().data;
        let sigmoid_value = 1.0 / (1.0 + (-data).exp());

        Self::new_with_children(
            sigmoid_value,
            format!("sigmoid({})", self.inner.borrow().label),
            Operation::Sigmoid(self.inner.clone()),
        )
    }

    pub fn relu(self) -> Self {
        Self::new_with_children(
            if self.inner.borrow().data < 0. {
                0.
            } else {
                self.inner.borrow().data
            },
            format!("ReLu({})", self.inner.borrow().label),
            Operation::ReLu(self.inner.clone()),
        )
    }

    pub fn backward(&self) {
        self.inner.borrow_mut().gradient = 1.;

        self.inner.borrow().backward();
    }

    pub fn value(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn nudge(&self, learning_rate: f64) {
        let grad = self.inner.borrow().gradient;
        let mut inner = self.inner.borrow_mut();

        inner.data -= learning_rate * grad;
        inner.gradient = 0.0;
    }

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.inner.borrow_mut().data = data;
    }
}

type SharedValueInner = Rc<RefCell<ValueInner>>;

#[derive(Debug)]
pub struct ValueInner {
    data: f64,
    label: String,
    op: Operation,
    gradient: f64,
}

impl ValueInner {
    fn backward(&self) {
        match &self.op {
            Operation::Const => (),
            Operation::Add(left, right) => {
                left.borrow_mut().gradient += self.gradient;
                right.borrow_mut().gradient += self.gradient;

                left.borrow().backward();
                right.borrow().backward();
            }
            Operation::Sub(left, right) => {
                left.borrow_mut().gradient += self.gradient;
                right.borrow_mut().gradient -= self.gradient;

                left.borrow().backward();
                right.borrow().backward();
            }
            Operation::Mul(left, right) => {
                left.borrow_mut().gradient += right.borrow().data * self.gradient;
                right.borrow_mut().gradient += left.borrow().data * self.gradient;

                left.borrow().backward();
                right.borrow().backward();
            }
            Operation::Pow(value, exp) => {
                let v = value.borrow().data;
                value.borrow_mut().gradient += (exp * v.powf(*exp - 1.0)) * self.gradient;

                value.borrow().backward();
            }
            Operation::Tanh(value) => {
                value.borrow_mut().gradient += (1.0 - self.data.powf(2.0)) * self.gradient;

                value.borrow().backward();
            }
            Operation::ReLu(value) => {
                value.borrow_mut().gradient += if self.data > 0. {
                    self.data * self.gradient
                } else {
                    0.
                };

                value.borrow().backward();
            }
            Operation::Sigmoid(value) => {
                let sigmoid_value = self.data; // this is σ(x)
                let sigmoid_derivative = sigmoid_value * (1.0 - sigmoid_value);
                value.borrow_mut().gradient += sigmoid_derivative * self.gradient;

                value.borrow().backward();
            }
        }
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new_with_children(
            self.inner.borrow().data + rhs.inner.borrow().data,
            format!(
                "{} + {}",
                self.inner.borrow().label,
                rhs.inner.borrow().label
            ),
            Operation::Add(self.inner.clone(), rhs.inner.clone()),
        )
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new_with_children(
            self.inner.borrow().data - rhs.inner.borrow().data,
            format!(
                "{} - {}",
                self.inner.borrow().label,
                rhs.inner.borrow().label
            ),
            Operation::Sub(self.inner.clone(), rhs.inner.clone()),
        )
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new_with_children(
            self.inner.borrow().data * rhs.inner.borrow().data,
            format!(
                "{} * {}",
                self.inner.borrow().label,
                rhs.inner.borrow().label
            ),
            Operation::Mul(self.inner.clone(), rhs.inner.clone()),
        )
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter()
            .fold(Self::new(0., "SUM"), |acc, value| acc + value)
    }
}

#[derive(Debug)]
enum Operation {
    Const,
    Add(SharedValueInner, SharedValueInner),
    Sub(SharedValueInner, SharedValueInner),
    Mul(SharedValueInner, SharedValueInner),
    Pow(SharedValueInner, f64),
    Tanh(SharedValueInner),
    ReLu(SharedValueInner),
    Sigmoid(SharedValueInner),
}
