use ndarray_linalg::*;
use ndarray::*;

use std::{
    f64,
    error::Error,
};

// only put the variables required to compute filters in the struct
// 1. 
#[allow(non_snake_case)]
pub struct GLM {

    // fixed
    pub T: Array2<f64>,
    pub H: Array2<f64>,
    pub Q: Array2<f64>,
    pub Z: Array2<f64>,
    pub R: Array2<f64>,
    pub y: Array3<f64>,

    // filters
}

#[allow(non_snake_case, dead_code)]
impl GLM {

    // create a new Gaussian Linear Model
    pub fn new(T: Array2<f64>, H: Array2<f64>, Q: Array2<f64>, Z: Array2<f64>, R: Array2<f64>, y:Array3<f64>) -> GLM {
        GLM {
            T: T,
            H: H,
            Q: Q,
            Z: Z,
            R: R,
            y: y
        }
    }

    // Run Kalman Filter on instance variables
    pub fn kalman_filter(&self) -> Result<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>), Box<dyn Error>> {

        let T = self.y.len();
        let mut axes_iterator: Array3<f64> = Array3::zeros((1, 1, T));

        let mut a_3d: Array3<f64> = Array3::zeros((2, 1, T));
        let mut v_3d: Array3<f64> = Array3::zeros((1, 1, T));
        let mut F_3d: Array3<f64> = Array3::zeros((1, 1, T));
        let mut P_3d: Array3<f64> = Array3::zeros((2, 2, T));
        let mut K_3d: Array3<f64> = Array3::zeros((2, 1, T));
        
        let mut a_prev: Array2<f64> = Array2::zeros((2, 1));
        let mut v_prev: Array2<f64> = Array2::zeros((1, 1));
        let mut F_prev: Array2<f64> = Array2::zeros((1, 1));
        let mut P_prev: Array2<f64> = Array2::zeros((2, 2));
        let mut K_prev: Array2<f64> = Array2::zeros((2, 1));

        // need to enumerate to use i
        for (i, _) in axes_iterator.axis_iter_mut(Axis(2)).enumerate() {
            
            // retrieve slices of the data
            let mut a_temp: ArrayViewMut2<f64> = a_3d.slice_mut(s![..,..,i]);
            let mut v_temp: ArrayViewMut2<f64> = v_3d.slice_mut(s![..,..,i]);
            let mut F_temp: ArrayViewMut2<f64> = F_3d.slice_mut(s![..,..,i]);
            let mut P_temp: ArrayViewMut2<f64> = P_3d.slice_mut(s![..,..,i]);
            let mut K_temp: ArrayViewMut2<f64> = K_3d.slice_mut(s![..,..,i]);
            

            // in first iteration: set first values of a and P and compute corresponding v and F
            // TODO: add diffuse initialization
            if i == 0 {

                // get y_0
                let y_temp: ArrayView2<f64> = self.y.slice(s![.., .., i]);

                // set a_0 and P_0
                a_temp.assign(&arr2(&[[0.0], [0.0]]));
                P_temp.assign(&arr2(&[[1.0, 0.0], [0.0, 1.0]]));

                // get first error and error variance: v and F
                v_temp.assign(&(
                    &y_temp - &self.Z.dot(&a_temp))
                );
                F_temp.assign(&(
                    &self.Z.dot(&P_temp.dot(&self.Z.t())) + &self.H
                ));

                // compute Kalman gain
                K_temp.assign(&(
                    &self.T.dot(
                        &P_temp.dot(
                            &self.Z.t().dot(
                                &F_temp.inv().unwrap()
                            )
                        )
                    )
                ));

                // persist lagged a in memory
                a_prev.assign(&a_temp);
                P_prev.assign(&P_temp);
                K_prev.assign(&K_temp);
                v_prev.assign(&v_temp);
                F_prev.assign(&F_temp);
            }

            else {

                // get new a_i and assign to mutable a_3d slice
                a_temp.assign(&(
                    &self.T.dot(&a_prev) + &K_prev.dot(&v_prev)
                ));

                // get new P_i and assign to mutable slide of P_3d
                P_temp.assign(&
                    (&self.T.dot(
                        &P_prev.dot(
                            &self.T.t()
                        )
                    ) + &self.R.dot(
                            &self.Q.dot(
                                &self.R.t()
                        )
                    ) + &K_prev.dot(
                            &F_prev.dot(
                                &K_prev.t()
                        )
                    ))
                ); 
                
                // get current y
                let y_temp: ArrayView2<f64> = self.y.slice(s![.., .., i]);
                
                // get prediction error
                v_temp.assign(&(
                    &y_temp - &self.Z.dot(&a_temp)
                ));

                // get prediction error variance
                F_temp.assign(&(
                    &self.Z.dot(&P_temp.dot(&self.Z.t())) + &self.H
                ));

                // compute Kalman gain
                K_temp.assign(&(
                    &self.T.dot(
                        &P_temp.dot(
                            &self.Z.t().dot(
                                &F_temp.inv().unwrap()
                            )
                        )
                    )
                ));

                // persist lagged a in memory
                a_prev.assign(&a_temp);
                P_prev.assign(&P_temp);
                K_prev.assign(&K_temp);
                v_prev.assign(&v_temp);
                F_prev.assign(&F_temp);
            }
        }

        Ok((a_3d, P_3d, v_3d, F_3d, K_3d))

    }

    // function that loads data
    pub fn print_shapes(&self) {
        print!("T shape: {:?} \n", self.T.shape());
        print!("H shape: {:?} \n", self.H.shape());
        print!("Q shape: {:?} \n", self.Q.shape());
        print!("Z shape: {:?} \n", self.Z.shape());
        print!("R shape: {:?} \n", self.R.shape());
        print!("y shape: {:?} \n", self.y.shape())
    }

}


