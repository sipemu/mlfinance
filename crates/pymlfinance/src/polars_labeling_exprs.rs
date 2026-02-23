use chrono::Utc;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;

use crate::polars_convert::*;
use crate::polars_kwargs::{TrendScanKwargs, VolKwargs};

fn map_err(e: impl std::fmt::Display) -> PolarsError {
    PolarsError::ComputeError(format!("{e}").into())
}

#[polars_expr(output_type=Float64)]
fn daily_volatility(inputs: &[Series], kwargs: VolKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let dummy_ts = vec![Utc::now(); values.len()];
    let result =
        mlfinance_labeling::volatility::daily_volatility(&values, &dummy_ts, kwargs.span as usize);
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Int32)]
fn trend_scanning_label_series(inputs: &[Series], kwargs: TrendScanKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_labeling::trend_scanning::trend_scanning_label_series(
        &values,
        kwargs.max_window as usize,
    )
    .map_err(map_err)?;
    Ok(make_output_i32(&inputs[0], result))
}
