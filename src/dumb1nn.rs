use rand::Rng;

pub fn model(x: f64, w: f64, b: f64) -> f64 {
    x * w + b
}

fn cost(w: f64, b: f64, td: &Vec<(f64, f64)>) -> f64 {
    let mut res = 0.0;
    for &(x, y) in td {
        let yh = model(x, w, b);
        let d = yh - y;
        res += d * d;
    }
    res / td.len() as f64
}

pub struct Params(f64, f64);
impl Params {
    pub fn w(&self) -> f64 {
        self.0
    }
    pub fn b(&self) -> f64 {
        self.1
    }
}

pub struct HyperParams {
    pub(crate) eps: f64,
    pub(crate) lr: f64,
    pub(crate) epochs: i32,
}

pub fn train(td: &Vec<(f64, f64)>, hp: &HyperParams, log: bool) -> Params {
    let mut rng = rand::thread_rng();
    let mut w: f64 = rng.gen_range(0.0..9.0);
    let mut b: f64 = rng.gen_range(0.0..3.0);

    for i in 0..hp.epochs {
        let c = cost(w, b, &td);
        let dc = (cost(w + hp.eps, b, &td) - c) / hp.eps;
        let db = (cost(w, b + hp.eps, &td) - c) / hp.eps;
        w -= dc * hp.lr;
        b -= db * hp.lr;
        let e = i + 1;
        if log {
            println!("[{e}] c = {c}; b = {b}; w = {w};");
        }
    }

    Params(w, b)
}
