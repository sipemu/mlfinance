use mlfinance::core::traits::Classifier;
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Bridge from a Python sklearn-like classifier to the Rust Classifier trait.
pub struct PyClassifier {
    obj: Py<PyAny>,
}

// SAFETY: Python GIL is acquired before every access to `obj`.
unsafe impl Send for PyClassifier {}
unsafe impl Sync for PyClassifier {}

impl PyClassifier {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }
}

impl Classifier for PyClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, sample_weight: Option<&Array1<f64>>) {
        Python::attach(|py| {
            let x_py = PyArray2::from_array(py, x);
            let y_py = PyArray1::from_array(py, y);
            let obj = self.obj.bind(py);

            let result = if let Some(w) = sample_weight {
                let w_py = PyArray1::from_array(py, w);
                let kwargs = PyDict::new(py);
                let _ = kwargs.set_item("sample_weight", w_py);
                obj.call_method("fit", (x_py, y_py), Some(&kwargs))
            } else {
                obj.call_method1("fit", (x_py, y_py))
            };

            if let Err(e) = result {
                eprintln!("PyClassifier.fit error: {e}");
            }
        });
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        Python::attach(|py| {
            let x_py = PyArray2::from_array(py, x);
            let obj = self.obj.bind(py);
            match obj.call_method1("predict", (x_py,)) {
                Ok(result) => {
                    if let Ok(arr) = result.cast::<PyArray1<f64>>() {
                        arr.to_vec()
                            .map(Array1::from_vec)
                            .unwrap_or_else(|_| Array1::zeros(0))
                    } else {
                        let v: Vec<f64> = result.extract().unwrap_or_default();
                        Array1::from_vec(v)
                    }
                }
                Err(e) => {
                    eprintln!("PyClassifier.predict error: {e}");
                    Array1::zeros(0)
                }
            }
        })
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        Python::attach(|py| {
            let x_py = PyArray2::from_array(py, x);
            let obj = self.obj.bind(py);
            match obj.call_method1("predict_proba", (x_py,)) {
                Ok(result) => {
                    if let Ok(arr) = result.cast::<PyArray2<f64>>() {
                        unsafe { arr.as_array().to_owned() }
                    } else {
                        Array2::zeros((0, 0))
                    }
                }
                Err(e) => {
                    eprintln!("PyClassifier.predict_proba error: {e}");
                    Array2::zeros((0, 0))
                }
            }
        })
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        Python::attach(|py| {
            let obj = self.obj.bind(py);
            match obj.getattr("feature_importances_") {
                Ok(attr) => {
                    if let Ok(arr) = attr.cast::<PyArray1<f64>>() {
                        arr.to_vec().ok().map(Array1::from_vec)
                    } else if let Ok(v) = attr.extract::<Vec<f64>>() {
                        Some(Array1::from_vec(v))
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        })
    }
}
