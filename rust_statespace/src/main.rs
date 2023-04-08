mod read_data;
mod glm;

use ndarray_linalg::*;
use ndarray::*;
use glm::GLM;

use std::{
    process,
    f64,
    error::Error,
};

fn main() {

    // create vectors
    let (x, y) = read_data::load_csv(String::from("./data/nile.csv")).unwrap_or_else(|err| {
        println!("Problem parsing data: {err}");
        process::exit(1);
    });

    let n = y.len();
    let y = y.into_shape((1, 1, n)).unwrap();

    // in reality will get these from Python
    let T = arr2(&[[1.0, 1.0], [0.0, 1.0]]);
    let H = arr2(&[[1.0]]);
    let Q = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let Z = arr2(&[[1.0, 0.0]]);
    let R = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

    let LLTM = GLM::new(T, H, Q, Z, R, y);
    LLTM.print_shapes();
    let (a_3d, P_3d, v_3d, F_3d, K_3d) = LLTM.kalman_filter().unwrap();

    println!("{}", &LLTM.y);
    println!("{}", a_3d.slice(s![0, .., ..]));

}
