use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;

use crate::polars_convert::*;
use crate::polars_kwargs::*;

#[polars_expr(output_type=Float64)]
fn amihud_lambda(inputs: &[Series]) -> PolarsResult<Series> {
    let returns = extract_f64(&inputs[0])?;
    let dollar_volumes = extract_f64(&inputs[1])?;
    let result =
        mlfinance_features::microstructure::amihud_lambda::amihud_lambda(&returns, &dollar_volumes);
    Ok(make_scalar(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn amihud_lambda_rolling(inputs: &[Series], kwargs: WindowKwargs) -> PolarsResult<Series> {
    let returns = extract_f64(&inputs[0])?;
    let dollar_volumes = extract_f64(&inputs[1])?;
    let n = returns.len();
    let result = mlfinance_features::microstructure::amihud_lambda::amihud_lambda_rolling(
        &returns,
        &dollar_volumes,
        kwargs.window as usize,
    );
    Ok(make_output(&inputs[0], pad_front(result, n)))
}

#[polars_expr(output_type=Float64)]
fn kyle_lambda(inputs: &[Series]) -> PolarsResult<Series> {
    let returns = extract_f64(&inputs[0])?;
    let signed_volume = extract_f64(&inputs[1])?;
    let result =
        mlfinance_features::microstructure::kyle_lambda::kyle_lambda(&returns, &signed_volume);
    Ok(make_scalar(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn roll_spread_rolling(inputs: &[Series], kwargs: WindowKwargs) -> PolarsResult<Series> {
    let prices = extract_f64(&inputs[0])?;
    let n = prices.len();
    let result = mlfinance_features::microstructure::roll_model::roll_spread_rolling(
        &prices,
        kwargs.window as usize,
    );
    Ok(make_output(&inputs[0], pad_front(result, n)))
}

#[polars_expr(output_type=Float64)]
fn corwin_schultz_spread(inputs: &[Series]) -> PolarsResult<Series> {
    let highs = extract_f64(&inputs[0])?;
    let lows = extract_f64(&inputs[1])?;
    let n = highs.len();
    let result =
        mlfinance_features::microstructure::corwin_schultz::corwin_schultz_spread(&highs, &lows);
    Ok(make_output(&inputs[0], pad_front(result, n)))
}

#[polars_expr(output_type=Float64)]
fn vpin(inputs: &[Series], kwargs: VpinKwargs) -> PolarsResult<Series> {
    let volumes = extract_f64(&inputs[0])?;
    let prices = extract_f64(&inputs[1])?;
    let result = mlfinance_features::microstructure::vpin::vpin(
        &volumes,
        &prices,
        kwargs.bucket_size,
        kwargs.n_buckets as usize,
    );
    Ok(make_output(&inputs[0], result))
}
