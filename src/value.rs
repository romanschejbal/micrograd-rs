use std::{
    cell::RefCell,
    collections::HashSet,
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
        // Build topological order
        let topo = build_topo(&self.inner);

        // Set root gradient to 1
        self.inner.borrow_mut().gradient = 1.;

        // Iterate in reverse topological order
        for node in topo.iter().rev() {
            let inner = node.borrow();
            let grad = inner.gradient;
            let data = inner.data;
            match &inner.op {
                Operation::Const => (),
                Operation::Add(left, right) => {
                    let left = Rc::clone(left);
                    let right = Rc::clone(right);
                    drop(inner);
                    left.borrow_mut().gradient += grad;
                    right.borrow_mut().gradient += grad;
                }
                Operation::Sub(left, right) => {
                    let left = Rc::clone(left);
                    let right = Rc::clone(right);
                    drop(inner);
                    left.borrow_mut().gradient += grad;
                    right.borrow_mut().gradient -= grad;
                }
                Operation::Mul(left, right) => {
                    let left_rc = Rc::clone(left);
                    let right_rc = Rc::clone(right);
                    let left_data = left.borrow().data;
                    let right_data = right.borrow().data;
                    drop(inner);
                    left_rc.borrow_mut().gradient += right_data * grad;
                    right_rc.borrow_mut().gradient += left_data * grad;
                }
                Operation::Pow(value, exp) => {
                    let value = Rc::clone(value);
                    let exp = *exp;
                    drop(inner);
                    let v = value.borrow().data;
                    value.borrow_mut().gradient += (exp * v.powf(exp - 1.0)) * grad;
                }
                Operation::Tanh(_value) => {
                    let value = match &node.borrow().op {
                        Operation::Tanh(v) => Rc::clone(v),
                        _ => unreachable!(),
                    };
                    value.borrow_mut().gradient += (1.0 - data.powf(2.0)) * grad;
                }
                Operation::ReLu(_value) => {
                    let value = match &node.borrow().op {
                        Operation::ReLu(v) => Rc::clone(v),
                        _ => unreachable!(),
                    };
                    value.borrow_mut().gradient += if data > 0. { grad } else { 0. };
                }
            }
        }
    }

    pub fn value(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn nudge(&self, learning_rate: f64) {
        let grad = self.inner.borrow().gradient;
        self.inner.borrow_mut().data -= learning_rate * grad;
    }

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.inner.borrow_mut().data = data;
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().gradient = 0.0;
    }

    pub fn gradient(&self) -> f64 {
        self.inner.borrow().gradient
    }

    pub fn add_gradient(&self, grad: f64) {
        self.inner.borrow_mut().gradient += grad;
    }
}

fn build_topo(root: &SharedValueInner) -> Vec<SharedValueInner> {
    let mut visited: HashSet<*const RefCell<ValueInner>> = HashSet::new();
    let mut topo: Vec<SharedValueInner> = Vec::new();

    fn visit(
        node: &SharedValueInner,
        visited: &mut HashSet<*const RefCell<ValueInner>>,
        topo: &mut Vec<SharedValueInner>,
    ) {
        let ptr = Rc::as_ptr(node);
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        let inner = node.borrow();
        match &inner.op {
            Operation::Const => (),
            Operation::Add(left, right)
            | Operation::Sub(left, right)
            | Operation::Mul(left, right) => {
                let left = Rc::clone(left);
                let right = Rc::clone(right);
                drop(inner);
                visit(&left, visited, topo);
                visit(&right, visited, topo);
            }
            Operation::Pow(value, _)
            | Operation::Tanh(value)
            | Operation::ReLu(value) => {
                let value = Rc::clone(value);
                drop(inner);
                visit(&value, visited, topo);
            }
        }

        topo.push(Rc::clone(node));
    }

    visit(root, &mut visited, &mut topo);
    topo
}

type SharedValueInner = Rc<RefCell<ValueInner>>;

#[derive(Debug)]
pub struct ValueInner {
    data: f64,
    label: String,
    op: Operation,
    gradient: f64,
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
}
