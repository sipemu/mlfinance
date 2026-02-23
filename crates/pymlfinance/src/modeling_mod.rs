use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::classifier::PyClassifier;
use crate::convert::*;
use crate::types::*;

// ── PurgedKFold class ───────────────────────────────────────────────────────

#[pyclass(name = "PurgedKFold")]
pub struct PyPurgedKFold {
    inner: mlfinance::modeling::cross_validation::purged_kfold::PurgedKFold,
}

#[pymethods]
impl PyPurgedKFold {
    #[new]
    #[pyo3(signature = (n_splits=5, embargo_pct=0.0))]
    fn new(n_splits: usize, embargo_pct: f64) -> Self {
        Self {
            inner: mlfinance::modeling::cross_validation::purged_kfold::PurgedKFold::new(
                n_splits,
                embargo_pct,
            ),
        }
    }

    fn split(&self, events: Vec<(usize, usize)>, n_samples: usize) -> Vec<PyFoldIndices> {
        let folds = self.inner.split(&events, n_samples);
        folds
            .into_iter()
            .map(|f| PyFoldIndices {
                train: f.train,
                test: f.test,
            })
            .collect()
    }
}

// ── Cross-validation scoring ────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (classifier, x, y, events, n_splits=5, embargo_pct=0.0, sample_weight=None, scoring=None))]
fn cv_score(
    py: Python<'_>,
    classifier: Py<PyAny>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    events: Vec<(usize, usize)>,
    n_splits: usize,
    embargo_pct: f64,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
    scoring: Option<Py<PyAny>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_arr = py_to_array2(x);
    let y_arr = py_to_array1(y);
    let sw = sample_weight.map(|w| py_to_array1(w));
    let mut clf = PyClassifier::new(classifier.clone_ref(py));

    let scoring_fn: Box<dyn Fn(&ndarray::Array1<f64>, &ndarray::Array1<f64>) -> f64> =
        if let Some(scorer) = scoring {
            let scorer = scorer.clone_ref(py);
            Box::new(
                move |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                    Python::attach(|py2| {
                        let yt = numpy::PyArray1::from_array(py2, y_true);
                        let yp = numpy::PyArray1::from_array(py2, y_pred);
                        scorer
                            .call1(py2, (yt, yp))
                            .and_then(|r| r.extract::<f64>(py2))
                            .unwrap_or(0.0)
                    })
                },
            )
        } else {
            Box::new(
                |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                    // Default: accuracy
                    let correct = y_true
                        .iter()
                        .zip(y_pred.iter())
                        .filter(|(&a, &b)| (a - b).abs() < 0.5)
                        .count();
                    correct as f64 / y_true.len().max(1) as f64
                },
            )
        };

    let scores = mlfinance::modeling::cross_validation::cv_score::cv_score(
        &mut clf,
        &x_arr,
        &y_arr,
        &events,
        n_splits,
        embargo_pct,
        sw.as_ref(),
        &*scoring_fn,
    );
    Ok(vec_to_py_array(py, scores))
}

// ── Feature importance ──────────────────────────────────────────────────────

#[pyfunction]
fn mean_decrease_impurity(
    py: Python<'_>,
    importances_per_tree: Vec<Vec<f64>>,
) -> Py<PyArray1<f64>> {
    let result =
        mlfinance::modeling::feature_importance::mdi::mean_decrease_impurity(&importances_per_tree);
    array1_to_py(py, result)
}

#[pyfunction]
#[pyo3(signature = (classifier, x, y, scoring=None, seed=42))]
fn mean_decrease_accuracy(
    py: Python<'_>,
    classifier: Py<PyAny>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    scoring: Option<Py<PyAny>>,
    seed: u64,
) -> Py<PyArray1<f64>> {
    let x_arr = py_to_array2(x);
    let y_arr = py_to_array1(y);
    let clf = PyClassifier::new(classifier.clone_ref(py));

    let scoring_fn = make_scoring_fn(py, scoring);

    let result = mlfinance::modeling::feature_importance::mda::mean_decrease_accuracy(
        &clf,
        &x_arr,
        &y_arr,
        &*scoring_fn,
        seed,
    );
    array1_to_py(py, result)
}

#[pyfunction]
#[pyo3(signature = (classifier, x, y, events, n_splits=5, scoring=None))]
fn single_feature_importance(
    py: Python<'_>,
    classifier: Py<PyAny>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    events: Vec<(usize, usize)>,
    n_splits: usize,
    scoring: Option<Py<PyAny>>,
) -> Py<PyArray1<f64>> {
    let x_arr = py_to_array2(x);
    let y_arr = py_to_array1(y);
    let mut clf = PyClassifier::new(classifier.clone_ref(py));

    let scoring_fn = make_scoring_fn(py, scoring);

    let result = mlfinance::modeling::feature_importance::sfi::single_feature_importance(
        &mut clf,
        &x_arr,
        &y_arr,
        &events,
        n_splits,
        &*scoring_fn,
    );
    vec_to_py_array(py, result)
}

#[pyfunction]
fn orthogonal_features(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    n_components: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let x_arr = py_to_array2(x);
    let (transformed, explained) = to_pyresult(
        mlfinance::modeling::feature_importance::orthogonal::orthogonal_features(
            &x_arr,
            n_components,
        ),
    )?;
    Ok((array2_to_py(py, transformed), array1_to_py(py, explained)))
}

#[pyfunction]
#[pyo3(signature = (x, y, weights=None))]
fn weighted_kendall_tau(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
) -> f64 {
    let xv = py_to_vec(x);
    let yv = py_to_vec(y);
    let wv = weights.map(|w| py_to_vec(w));
    mlfinance::modeling::feature_importance::kendall_tau::weighted_kendall_tau(
        &xv,
        &yv,
        wv.as_deref(),
    )
}

#[pyfunction]
fn make_classification(
    py: Python<'_>,
    n_samples: usize,
    n_informative: usize,
    n_redundant: usize,
    n_noise: usize,
    seed: u64,
) -> (Py<PyArray2<f64>>, Py<PyArray1<f64>>) {
    let result = mlfinance::modeling::feature_importance::synthetic_data::make_classification(
        n_samples,
        n_informative,
        n_redundant,
        n_noise,
        seed,
    );
    (array2_to_py(py, result.x), array1_to_py(py, result.y))
}

// ── Hyperparameter search ───────────────────────────────────────────────────

#[pyfunction]
fn grid_search(
    py: Python<'_>,
    param_grids: Vec<(String, Vec<f64>)>,
    score_fn: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    use mlfinance::modeling::hyperparams::grid_search::{self as gs, ParamGrid};

    let grids: Vec<ParamGrid> = param_grids
        .into_iter()
        .map(|(name, values)| ParamGrid { name, values })
        .collect();

    let score_fn_ref = score_fn.clone_ref(py);
    let scoring = move |params: &[(String, f64)]| -> f64 {
        Python::attach(|py2| {
            let dict = pyo3::types::PyDict::new(py2);
            for (k, v) in params {
                let _ = dict.set_item(k, v);
            }
            score_fn_ref
                .call1(py2, (dict,))
                .and_then(|r| r.extract::<f64>(py2))
                .unwrap_or(f64::NEG_INFINITY)
        })
    };

    let result = gs::grid_search(&grids, &scoring);
    let dict = pyo3::types::PyDict::new(py);
    let best_dict = pyo3::types::PyDict::new(py);
    for (k, v) in &result.best_params {
        best_dict.set_item(k, v)?;
    }
    dict.set_item("best_params", best_dict)?;
    dict.set_item("best_score", result.best_score)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
fn random_search(
    py: Python<'_>,
    param_distributions: Vec<(String, f64, f64)>,
    n_iter: usize,
    score_fn: Py<PyAny>,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let score_fn_ref = score_fn.clone_ref(py);
    let scoring = move |params: &[(String, f64)]| -> f64 {
        Python::attach(|py2| {
            let dict = pyo3::types::PyDict::new(py2);
            for (k, v) in params {
                let _ = dict.set_item(k, v);
            }
            score_fn_ref
                .call1(py2, (dict,))
                .and_then(|r| r.extract::<f64>(py2))
                .unwrap_or(f64::NEG_INFINITY)
        })
    };

    let result = mlfinance::modeling::hyperparams::random_search::random_search(
        &param_distributions,
        n_iter,
        &scoring,
        seed,
    );
    let dict = pyo3::types::PyDict::new(py);
    let best_dict = pyo3::types::PyDict::new(py);
    for (k, v) in &result.best_params {
        best_dict.set_item(k, v)?;
    }
    dict.set_item("best_params", best_dict)?;
    dict.set_item("best_score", result.best_score)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
fn log_uniform_sample(
    py: Python<'_>,
    low: f64,
    high: f64,
    n: usize,
    seed: u64,
) -> Py<PyArray1<f64>> {
    let result =
        mlfinance::modeling::hyperparams::log_uniform::log_uniform_sample(low, high, n, seed);
    vec_to_py_array(py, result)
}

// ── Scoring functions ───────────────────────────────────────────────────────

#[pyfunction]
fn f1_score(y_true: PyReadonlyArray1<'_, f64>, y_pred: PyReadonlyArray1<'_, f64>) -> f64 {
    let yt = py_to_vec(y_true);
    let yp = py_to_vec(y_pred);
    mlfinance::modeling::hyperparams::scoring::f1_score(&yt, &yp)
}

#[pyfunction]
fn neg_log_loss(y_true: PyReadonlyArray1<'_, f64>, y_proba: PyReadonlyArray1<'_, f64>) -> f64 {
    let yt = py_to_vec(y_true);
    let yp = py_to_vec(y_proba);
    mlfinance::modeling::hyperparams::scoring::neg_log_loss(&yt, &yp)
}

#[pyfunction]
fn accuracy_score(y_true: PyReadonlyArray1<'_, f64>, y_pred: PyReadonlyArray1<'_, f64>) -> f64 {
    let yt = py_to_vec(y_true);
    let yp = py_to_vec(y_pred);
    mlfinance::modeling::hyperparams::scoring::accuracy_score(&yt, &yp)
}

// ── Ensemble ────────────────────────────────────────────────────────────────

#[pyfunction]
fn bagging_accuracy(n: usize, p: f64) -> f64 {
    mlfinance::modeling::ensemble::bagging::bagging_accuracy(n, p)
}

// ── Helper ──────────────────────────────────────────────────────────────────

fn make_scoring_fn(
    py: Python<'_>,
    scoring: Option<Py<PyAny>>,
) -> Box<dyn Fn(&ndarray::Array1<f64>, &ndarray::Array1<f64>) -> f64> {
    if let Some(scorer) = scoring {
        let scorer = scorer.clone_ref(py);
        Box::new(
            move |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                Python::attach(|py2| {
                    let yt = numpy::PyArray1::from_array(py2, y_true);
                    let yp = numpy::PyArray1::from_array(py2, y_pred);
                    scorer
                        .call1(py2, (yt, yp))
                        .and_then(|r| r.extract::<f64>(py2))
                        .unwrap_or(0.0)
                })
            },
        )
    } else {
        Box::new(
            |y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>| {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&a, &b)| (a - b).abs() < 0.5)
                    .count();
                correct as f64 / y_true.len().max(1) as f64
            },
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPurgedKFold>()?;
    m.add_function(wrap_pyfunction!(cv_score, m)?)?;
    m.add_function(wrap_pyfunction!(mean_decrease_impurity, m)?)?;
    m.add_function(wrap_pyfunction!(mean_decrease_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(single_feature_importance, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_features, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_kendall_tau, m)?)?;
    m.add_function(wrap_pyfunction!(make_classification, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search, m)?)?;
    m.add_function(wrap_pyfunction!(random_search, m)?)?;
    m.add_function(wrap_pyfunction!(log_uniform_sample, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score, m)?)?;
    m.add_function(wrap_pyfunction!(neg_log_loss, m)?)?;
    m.add_function(wrap_pyfunction!(accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(bagging_accuracy, m)?)?;
    Ok(())
}
