use ndarray_linalg::*;
use ndarray::*;

use std::{
    f64,
    error::Error,
};

// only put the variables required to compute filters in the struct
#[allow(non_snake_case)]
pub struct GLM {

    // fixed
    pub T: Array2<f64>,
    pub H: Array2<f64>,
    pub Q: Array2<f64>,
    pub Z: Array2<f64>,
    pub R: Array2<f64>,
    pub y: Array3<f64>,

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
    pub fn kalman_filter(&self) -> Result<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>), Box<dyn Error>> {

        let T = self.y.len();
        let p = self.T.ncols();
        let s: usize = self.Z.nrows();

        let mut axes_iterator: Array3<f64> = Array3::zeros((1, 1, T));

        let mut a_3d:   Array3<f64> = Array3::zeros((p, 1, T));
        let mut att_3d: Array3<f64> = Array3::zeros((p, 1, T));
        let mut v_3d:   Array3<f64> = Array3::zeros((s, 1, T));
        let mut F_3d:   Array3<f64> = Array3::zeros((s, s, T));
        let mut P_3d:   Array3<f64> = Array3::zeros((p, p, T));
        let mut Ptt_3d: Array3<f64> = Array3::zeros((p, p, T));
        let mut M_3d:   Array3<f64> = Array3::zeros((p, s, T));
        let mut K_3d:   Array3<f64> = Array3::zeros((p, s, T));
        
        let mut a_prev:   Array2<f64> = Array2::zeros((p, 1));
        let mut att_prev: Array2<f64> = Array2::zeros((p, 1));
        let mut v_prev:   Array2<f64> = Array2::zeros((s, 1));
        let mut F_prev:   Array2<f64> = Array2::zeros((s, s));
        let mut P_prev:   Array2<f64> = Array2::zeros((p, p));
        let mut Ptt_prev: Array2<f64> = Array2::zeros((p, p));
        let mut M_prev:   Array2<f64> = Array2::zeros((p, s));
        let mut K_prev:   Array2<f64> = Array2::zeros((p, s));

        // need to enumerate to use i
        for (i, _) in axes_iterator.axis_iter_mut(Axis(2)).enumerate() {
            
            // retrieve slices of the data
            let mut a_temp:   ArrayViewMut2<f64> = a_3d.slice_mut(s![..,..,i]);
            let mut att_temp: ArrayViewMut2<f64> = att_3d.slice_mut(s![..,..,i]);
            let mut v_temp:   ArrayViewMut2<f64> = v_3d.slice_mut(s![..,..,i]);
            let mut F_temp:   ArrayViewMut2<f64> = F_3d.slice_mut(s![..,..,i]);
            let mut P_temp:   ArrayViewMut2<f64> = P_3d.slice_mut(s![..,..,i]);
            let mut Ptt_temp: ArrayViewMut2<f64> = Ptt_3d.slice_mut(s![..,..,i]);
            let mut M_temp:   ArrayViewMut2<f64> = M_3d.slice_mut(s![..,..,i]);
            let mut K_temp:   ArrayViewMut2<f64> = K_3d.slice_mut(s![..,..,i]);
            
            // in first iteration: set first values of a and P and compute corresponding v and F
            // TODO: add diffuse initialization
            // TODO: add incasting
            if i == 0 {

                // set a_0 and P_0
                a_temp.assign(&Array2::zeros((p, 1)));
                P_temp.assign(&(1e6 * &Array2::eye(p)));

                // get y_0
                let y_temp: ArrayView2<f64> = self.y.slice(s![.., .., i]);

                // get first error and error variance: v and F
                v_temp.assign(&(
                    &y_temp - &self.Z.dot(&a_temp))
                );

                F_temp.assign(&(
                    &self.Z.dot(&P_temp.dot(&self.Z.t())) + &self.H
                ));

                // compute incasted Kalman gain M
                M_temp.assign(&(
                    &P_temp.dot(
                        &self.Z.t().dot(
                            &F_temp.inv().unwrap()
                        )
                    )
                ));

                // compute Kalman gain K
                K_temp.assign(&(
                    &self.T.dot(
                        &M_temp)
                ));
                
                // compute incasted att
                att_temp.assign(&(
                    &a_temp + &M_temp.dot(&v_temp)
                ));

                // compute incasted Ptt
                Ptt_temp.assign(&(
                    &P_temp - 
                        &M_temp.dot(&F_temp.dot(&M_temp.t()))
                ));

                // persist lagged a in memory
                a_prev.assign(&a_temp);
                P_prev.assign(&P_temp);
                M_prev.assign(&M_temp);
                K_prev.assign(&K_temp);
                v_prev.assign(&v_temp);
                F_prev.assign(&F_temp);
                att_prev.assign(&att_temp);
                Ptt_prev.assign(&Ptt_temp);
            }

            else {

                // get new a_i and assign to mutable a_3d slice
                a_temp.assign(&(
                    &self.T.dot(&att_prev)
                ));

                // get new P_i and assign to mutable slide of P_3d
                P_temp.assign(&
                    (&self.T.dot(
                        &Ptt_prev.dot(
                            &self.T.t()
                        )
                    ) + &self.R.dot(
                            &self.Q.dot(
                                &self.R.t()
                            )
                        ) 
                    )
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

                // compute incasted Kalman gain M
                M_temp.assign(&(
                    &P_temp.dot(
                        &self.Z.t().dot(
                            &F_temp.inv().unwrap()
                        )
                    )
                ));

                // compute Kalman gain K
                K_temp.assign(&(
                    &self.T.dot(
                        &M_temp)
                ));

                // compute incasted att
                att_temp.assign(&(
                    &a_temp + &M_temp.dot(&v_temp)
                ));

                // compute incasted Ptt
                Ptt_temp.assign(&(
                    &P_temp - 
                        &M_temp.dot(&F_temp.dot(&M_temp.t()))
                ));

                // persist lagged a in memory
                a_prev.assign(&a_temp);
                P_prev.assign(&P_temp);
                M_prev.assign(&M_temp);
                K_prev.assign(&K_temp);
                v_prev.assign(&v_temp);
                F_prev.assign(&F_temp);
                att_prev.assign(&att_temp);
                Ptt_prev.assign(&Ptt_temp);
            }
        }

        Ok((a_3d, att_3d, P_3d, Ptt_3d, v_3d, F_3d, K_3d, M_3d))

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


