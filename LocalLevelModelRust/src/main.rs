// TO DO next iteration
// 1. Read Nile Data
// 2. Plot data w/ filters
// 3. Test log likelihood
// 4. 

use ndarray::{Array1, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use num_traits::Float;

#[allow(non_snake_case)]
fn main() {

    // init matrices
    let T: f64 = 1.0;
    let Z: f64 = 1.0;
    let R: f64 = 1.0;
    let Q: f64 = 1.0;
    let H: f64 = 1.0;

    // create vectors
    let y = Array1::random(10, Uniform::new(0., 10.));
    let nile = 
    print_type_of(&y);

    // define shapes
    //let n = y.len();
    let n = y.len() as u32;
    let p: u32 = 1;
    let s: u32 = 1;

    // initialize local level model

    // run local level model
    let (a, P, v, F) = kalman_filter(&T, &Z, &R, &Q, &H, &y, &n);
    let llik = log_likelihood(&v, &F);

    // print results
    println!("Kalman filter output: {:?}", a);
    println!("Log likelihood: {:?}", llik);
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
fn log_likelihood(v: &Array1<f64>, F: &Array1<f64>) -> f64 {

    let n: f64 = v.len() as f64;
    let pi = std::f64::consts::PI;
    let part_1: f64 = - (n / 2.0) * (2.0 * pi).ln();
    let part_2: f64 = - (1.0 / 2.0) * F.map(|x| x.ln()).sum();
    let part_3: f64 = - (1.0 / 2.0) * ((v * v) / F).sum();

    part_1 + part_2 + part_3
}


