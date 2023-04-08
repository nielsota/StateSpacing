mod glm;

use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyArray3};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn rust_statespace(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // wrapper of `axpy`
    #[allow(non_snake_case)]
    #[pyfn(m)]
    #[pyo3(name = "kalman_filter")]
    fn kalman_filter_py<'py>(
        py: Python<'py>,
        T: PyReadonlyArray2<f64>,
        H: PyReadonlyArray2<f64>,
        Q: PyReadonlyArray2<f64>,
        Z: PyReadonlyArray2<f64>,
        R: PyReadonlyArray2<f64>,
        y: PyReadonlyArray3<f64>,
    ) -> (&'py PyArray3<f64>, 
    &'py PyArray3<f64>,
    &'py PyArray3<f64>,
    &'py PyArray3<f64>,
    &'py PyArray3<f64>)
     {
        
        let T = T.as_array().to_owned();
        let H = H.as_array().to_owned();
        let Q = Q.as_array().to_owned();
        let Z = Z.as_array().to_owned();
        let R = R.as_array().to_owned();
        let y = y.as_array().to_owned();

        let LLTM = glm::GLM::new(T, H, Q, Z, R, y);

        let (a_3d, P_3d, v_3d, F_3d, K_3d) = LLTM.kalman_filter().unwrap();

        ((a_3d).into_pyarray(py), (P_3d).into_pyarray(py), (v_3d).into_pyarray(py), (F_3d).into_pyarray(py), (K_3d).into_pyarray(py))
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