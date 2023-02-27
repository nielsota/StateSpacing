// TO DO next iteration
// 1. Read Nile Data DONE
// 2. Plot data w/ filters
// 3. Test log likelihood  DONE
// 4. Make struct for system matrices

use std::{
    env,
    error::Error,
    ffi::OsString,
    fs::File,
    process,
    f64
};

use csv::StringRecord;

use ndarray::{Array1, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use num_traits::Float;
use plotters::prelude::*;

mod read_data;

#[allow(non_snake_case)]
fn main() {

    // init matrices
    let T: f64 = 1.0;
    let Z: f64 = 1.0;
    let R: f64 = 1.0;
    let Q: f64 = 1.0;
    let H: f64 = 1.0;

    // create vectors
    let (x, y) = read_data::load_csv(String::from("./data/nile.csv")).unwrap_or_else(|err| {
        println!("Problem parsing data: {err}");
        process::exit(1);
    });;

    //let x_min = (&x).min() as i32

    // plotting setup: 
    let data: Vec<(i32, f64)>= x.map(|x_i| *x_i as i32).iter().cloned().zip(y.iter().cloned()).collect();
    let root_area = BitMapBackend::new("./images/test.png", (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();
    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40.0)
        .set_label_area_size(LabelAreaPosition::Bottom, 40.0)
        .set_label_area_size(LabelAreaPosition::Right, 40.0)
        .set_label_area_size(LabelAreaPosition::Top, 40.0)
        .caption("Nile Data", ("sans-serif", 40.0))
        .build_cartesian_2d(1800..1950, 0.0..1000.0)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    // Draw Scatter Plot: https://towardsdatascience.com/how-to-create-plot-in-rust-fdc6c024461c
    ctx.draw_series(
        data.iter().map(|point| Circle::new(*point, 4.0_f64, &BLUE)),
    ).unwrap();

    // define shapes
    //let n = y.len();
    let n = y.len() as u32;
    let p: u32 = 1;
    let s: u32 = 1;

    // initialize local level model

    // run local level model
    let (a, P, v, F) = kalman_filter(&T, &Z, &R, &Q, &H, &y, &n);
    let llik = log_likelihood(&v, &F).unwrap_or_else(|err| {
        println!("Problem computing log likelihood: {err}");
        process::exit(1);
    });

    // print results
    //println!("Kalman filter output: {:?}", a);
    //println!("Log likelihood: {:?}", llik);
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

#[allow(non_snake_case)]
fn kalman_filter(T: &f64, Z: &f64, R: &f64, Q: &f64, H: &f64, y: &Array1<f64>, n: &u32) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)
{
    println!("T: {}", T);
    println!("Z: {}", Z);
    println!("R: {}", R);
    println!("Q: {}", Q);
    println!("H: {}", H);

    println!("y shape: {}", y.len());
    println!("y: {}", y);

    // create return vectors
    let mut a = vec![];
    let mut P = vec![];
    let mut v = vec![];
    let mut F = vec![];

    a.push(0 as f64);
    P.push(10000 as f64);

    for t in 0..y.len() {

        // perform kalman updates
        let v_temp = y[t] - a[t];
        let F_temp = P[t] + H;
        let k_temp = P[t] / F_temp;
        let a_temp = a[t] + k_temp * v_temp;
        let P_temp = k_temp * H + Q;
        
        // push into vectors
        v.push(v_temp);
        F.push(F_temp);
        a.push(a_temp);
        P.push(P_temp);

    }

    // return tuple containing filter output
    (Array1::from(a), Array1::from(P), Array1::from(v), Array1::from(F))
}

#[allow(non_snake_case)]
fn log_likelihood(v: &Array1<f64>, F: &Array1<f64>) -> Result<f64, Box<dyn Error>> {

    let n: f64 = v.len() as f64;
    let pi = std::f64::consts::PI;
    let part_1: f64 = - (n / 2.0) * (2.0 * pi).ln();
    let part_2: f64 = - (1.0 / 2.0) * F.map(|x| x.ln()).sum();
    let part_3: f64 = - (1.0 / 2.0) * ((v * v) / F).sum();

    Ok(part_1 + part_2 + part_3)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llik_simple() {
        // Back of the envelope developed in airplane. -> make more advanced later.
        let v = vec![0.0, 0.0];
        let F = vec![1.0, 1.0];
        let pi = std::f64::consts::PI;
        assert_eq!(-(2.0*pi).ln(), log_likelihood(&Array1::from(v), &Array1::from(F)).unwrap());
    }
}

