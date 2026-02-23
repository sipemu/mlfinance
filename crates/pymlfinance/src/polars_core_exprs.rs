use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;

use crate::polars_convert::*;
use crate::polars_kwargs::EwmaKwargs;

fn map_err(e: impl std::fmt::Display) -> PolarsError {
    PolarsError::ComputeError(format!("{e}").into())
}

#[polars_expr(output_type=Float64)]
fn ewma(inputs: &[Series], kwargs: EwmaKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_core::math::ewma(&values, kwargs.span as usize).map_err(map_err)?;
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn ewma_std(inputs: &[Series], kwargs: EwmaKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_core::math::ewma_std(&values, kwargs.span as usize).map_err(map_err)?;
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn cumsum(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_core::math::cumsum(&values);
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn log_returns(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let n = values.len();
    let result = mlfinance_core::math::log_returns(&values);
    Ok(make_output(&inputs[0], pad_front(result, n)))
}

#[polars_expr(output_type=Float64)]
fn simple_returns(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let n = values.len();
    let result = mlfinance_core::math::simple_returns(&values);
    Ok(make_output(&inputs[0], pad_front(result, n)))
}
