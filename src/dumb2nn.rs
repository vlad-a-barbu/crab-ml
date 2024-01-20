use rand::Rng;

pub fn model(x: [f64; 2], w: [f64; 2], b: f64) -> f64 {
    x[0] * w[0] + x[1] * w[1] + b
}

fn cost(
    w: [f64; 2],
    b: f64,
    td: &Vec<(f64, f64, f64)>,
    act: &Box<dyn Fn(f64) -> f64 + Send + Sync>,
) -> f64 {
    let mut res = 0.0;
    for &(x1, x2, y) in td {
        let yh = act(model([x1, x2], w, b));
        let d = yh - y;
        res += d * d;
    }
    res / td.len() as f64
}

pub struct Params([f64; 2], f64);
impl Params {
    pub fn w(&self) -> [f64; 2] {
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
    pub(crate) act: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    pub(crate) wrange: (f64, f64),
    pub(crate) brange: (f64, f64),
}

pub fn train(td: &Vec<(f64, f64, f64)>, hp: &HyperParams, log: bool) -> Params {
    let mut rng = rand::thread_rng();
    let mut w1: f64 = rng.gen_range(hp.wrange.0..hp.wrange.1);
    let mut w2: f64 = rng.gen_range(hp.wrange.0..hp.wrange.1);
    let w = [w1, w2];
    let mut b: f64 = rng.gen_range(hp.brange.0..hp.brange.1);

    for i in 0..hp.epochs {
        let c = cost(w, b, &td, &hp.act);
        let dc1 = (cost([w1 + hp.eps, w2], b, &td, &hp.act) - c) / hp.eps;
        let dc2 = (cost([w1, w2 + hp.eps], b, &td, &hp.act) - c) / hp.eps;
        let db = (cost(w, b + hp.eps, &td, &hp.act) - c) / hp.eps;
        w1 -= dc1 * hp.lr;
        w2 -= dc2 * hp.lr;
        b -= db * hp.lr;
        let e = i + 1;
        if log {
            println!("[{e}] c = {c}; b = {b}; w1 = {w1}; w2 = {w2};");
        }
    }

    Params(w, b)
}
