use std::f64::consts::E;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}
