use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;

use crate::polars_convert::*;
use crate::polars_kwargs::*;

// -- Structural breaks --

#[polars_expr(output_type=Float64)]
fn adf_test(inputs: &[Series], kwargs: AdfKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let (t_stat, _betas) =
        mlfinance_features::structural_breaks::adf::adf_test(&values, kwargs.max_lags as usize);
    Ok(make_scalar(&inputs[0], t_stat))
}

#[polars_expr(output_type=Float64)]
fn sadf(inputs: &[Series], kwargs: SadfKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let n = values.len();
    let result = mlfinance_features::structural_breaks::sadf::sadf(
        &values,
        kwargs.min_window as usize,
        kwargs.max_lags as usize,
    );
    Ok(make_output(&inputs[0], pad_front(result, n)))
}

// -- Encoding --

#[polars_expr(output_type=Int32)]
fn binary_encode(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_features::entropy::encoding::binary_encode(&values);
    let int_result: Vec<i32> = result.iter().map(|&b| if b { 1 } else { 0 }).collect();
    Ok(make_output_i32(&inputs[0], int_result))
}

#[polars_expr(output_type=Int32)]
fn quantile_encode(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result =
        mlfinance_features::entropy::encoding::quantile_encode(&values, kwargs.n_bins as usize);
    let int_result: Vec<i32> = result.iter().map(|&v| v as i32).collect();
    Ok(make_output_i32(&inputs[0], int_result))
}

#[polars_expr(output_type=Int32)]
fn sigma_encode(inputs: &[Series], kwargs: SigmaKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result =
        mlfinance_features::entropy::encoding::sigma_encode(&values, kwargs.n_bands as usize);
    let int_result: Vec<i32> = result.iter().map(|&v| v as i32).collect();
    Ok(make_output_i32(&inputs[0], int_result))
}

// -- Entropy --

#[polars_expr(output_type=Float64)]
fn lempel_ziv_complexity(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let binary = mlfinance_features::entropy::encoding::binary_encode(&values);
    let result = mlfinance_features::entropy::lempel_ziv::lempel_ziv_complexity(&binary);
    Ok(make_scalar(&inputs[0], result as f64))
}

#[polars_expr(output_type=Float64)]
fn kontoyiannis_entropy(inputs: &[Series], kwargs: EntropyKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let symbols = mlfinance_features::entropy::encoding::quantile_encode(&values, 10);
    let result = mlfinance_features::entropy::kontoyiannis::kontoyiannis_entropy(
        &symbols,
        kwargs.window as usize,
    );
    Ok(make_scalar(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn shannon_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_features::entropy::shannon::shannon_entropy(&values);
    Ok(make_scalar(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn plugin_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let n_symbols = 10;
    let symbols = mlfinance_features::entropy::encoding::quantile_encode(&values, n_symbols);
    let result = mlfinance_features::entropy::plugin::plugin_entropy(&symbols, n_symbols);
    Ok(make_scalar(&inputs[0], result))
}

// -- Microstructure (single-column) --

#[polars_expr(output_type=Float64)]
fn tick_rule_classify(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_features::microstructure::tick_rule::tick_rule_classify(&values);
    Ok(make_output(&inputs[0], result))
}
