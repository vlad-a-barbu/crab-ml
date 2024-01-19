mod dumbnn;

fn main() {
    let td: Vec<(f64, f64)> = (1..30).map(|x| (x as f64, (x * 2) as f64)).collect();
    let params = dumbnn::train(&td, false);
    let w = params.w();
    let b = params.b();
    println!("w = {w}; b = {b};");

    let vd: Vec<(f64, f64)> = (30..=33).map(|x| (x as f64, (x * 2) as f64)).collect();
    for (x, y) in vd {
        let yh = dumbnn::model(x, w, b);
        println!("expected = {y}; actual = {yh};")
    }
}
