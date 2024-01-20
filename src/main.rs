mod activations;
mod dumb1nn;
mod dumb2nn;

fn eval_dumb1(hp: &dumb1nn::HyperParams, log_train: bool) {
    let td: Vec<(f64, f64)> = (1..30).map(|x| (x as f64, (x * 2) as f64)).collect();
    let params = dumb1nn::train(&td, &hp, log_train);
    let w = params.w();
    let b = params.b();
    println!("w = {w}; b = {b};");

    let vd: Vec<(f64, f64)> = (30..=33).map(|x| (x as f64, (x * 2) as f64)).collect();
    for (x, y) in vd {
        let yh = dumb1nn::model(x, w, b);
        println!("expected = {y}; actual = {yh};")
    }
}

fn eval_dumb2(hp: &dumb2nn::HyperParams, log_train: bool) {
    let or: Vec<(f64, f64, f64)> = Vec::from([
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ]);
    let and: Vec<(f64, f64, f64)> = Vec::from([
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    ]);
    let nand: Vec<(f64, f64, f64)> = Vec::from([
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
    ]);
    for (td, name) in [(or, "OR"), (and, "AND"), (nand, "NAND")] {
        println!("\n{name}");
        let params = dumb2nn::train(&td, &hp, log_train);
        let (w1, w2) = (params.w()[0], params.w()[1]);
        let b = params.b();
        println!("w1 = {w1}; w2 = {w2}; b = {b};");

        for (x1, x2, y) in td {
            let yh = activations::sigmoid(dumb2nn::model([x1, x2], [w1, w2], b));
            println!("expected = {y}; actual = {yh};")
        }
    }
}

fn main() {
    let hp = dumb2nn::HyperParams {
        eps: 1e-1,
        lr: 1e-1,
        epochs: 1_000_000,
        act: Box::new(activations::sigmoid),
        wrange: (0.0, 10.0),
        brange: (0.0, 1.0),
    };
    eval_dumb2(&hp, false);
}
