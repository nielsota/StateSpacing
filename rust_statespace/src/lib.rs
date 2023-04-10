mod glm;

use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyArray2, PyArray3};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use glm::GLM;

#[pymodule]
fn rust_statespace(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // wrapper of kalman filter
    #[allow(non_snake_case)]
    #[pyfn(m)]
    #[pyo3(name = "kalman_filter")]
    pub fn kalman_filter_py<'py>(
        py: Python<'py>,
        T: PyReadonlyArray2<f64>,
        H: PyReadonlyArray2<f64>,
        Q: PyReadonlyArray2<f64>,
        Z: PyReadonlyArray2<f64>,
        R: PyReadonlyArray2<f64>,
        y: PyReadonlyArray3<f64>,
    ) -> 
    (&'py PyArray3<f64>, 
    &'py PyArray3<f64>,
    &'py PyArray3<f64>,
    &'py PyArray3<f64>,
    &'py PyArray3<f64>,
    &'py PyArray3<f64>,
    &'py PyArray3<f64>)
     {
        
        // get owned representations of the data
        let T = T.as_array().to_owned();
        let H = H.as_array().to_owned();
        let Q = Q.as_array().to_owned();
        let Z = Z.as_array().to_owned();
        let R = R.as_array().to_owned();
        let y = y.as_array().to_owned();

        // instantiate the GLM
        let LLTM = GLM::new(T, H, Q, Z, R, y);

        // run the kalman filter
        let (a_3d, att_3d, P_3d, Ptt_3d, v_3d, F_3d, K_3d) = LLTM.kalman_filter().unwrap();

        // return the arrays as numpy arrays
        ((a_3d).into_pyarray(py), (att_3d).into_pyarray(py), (P_3d).into_pyarray(py), (Ptt_3d).into_pyarray(py), (v_3d).into_pyarray(py), (F_3d).into_pyarray(py), (K_3d).into_pyarray(py))
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_something() {
        assert_eq!(4, 2+2)
    }
}