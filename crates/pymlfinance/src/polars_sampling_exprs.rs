use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;

use crate::polars_convert::*;
use crate::polars_kwargs::{FindMinDKwargs, FracDiffKwargs};

#[polars_expr(output_type=Float64)]
fn frac_diff_ffd(inputs: &[Series], kwargs: FracDiffKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let n = values.len();
    let result =
        mlfinance_sampling::fracdiff::ffd::frac_diff_ffd(&values, kwargs.d, kwargs.threshold);
    Ok(make_output(&inputs[0], pad_front(result, n)))
}

#[polars_expr(output_type=Float64)]
fn frac_diff_expanding(inputs: &[Series], kwargs: FracDiffKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let n = values.len();
    let result = mlfinance_sampling::fracdiff::expanding::frac_diff_expanding(
        &values,
        kwargs.d,
        kwargs.threshold,
    );
    Ok(make_output(&inputs[0], pad_front(result, n)))
}

#[polars_expr(output_type=Float64)]
fn find_min_d(inputs: &[Series], kwargs: FindMinDKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_sampling::fracdiff::min_d::find_min_d(
        &values,
        kwargs.max_d,
        kwargs.step_size,
        kwargs.threshold,
    );
    Ok(make_scalar(&inputs[0], result))
}
