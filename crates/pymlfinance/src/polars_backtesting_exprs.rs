use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;

use crate::polars_convert::*;
use crate::polars_kwargs::*;

#[polars_expr(output_type=Float64)]
fn sharpe_ratio(inputs: &[Series], kwargs: SharpeKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_backtesting::statistics::sharpe::sharpe_ratio(
        &values,
        kwargs.risk_free_rate,
        kwargs.periods_per_year,
    );
    Ok(make_scalar(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn hit_ratio(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_backtesting::statistics::general::hit_ratio(&values);
    Ok(make_scalar(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn hhi(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_backtesting::statistics::hhi::hhi(&values);
    Ok(make_scalar(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn compute_drawdowns(inputs: &[Series]) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result = mlfinance_backtesting::statistics::drawdown::compute_drawdowns(&values);
    Ok(make_output(&inputs[0], result.drawdown_series))
}

#[polars_expr(output_type=Float64)]
fn sigmoid_bet_size(inputs: &[Series], kwargs: BetSizeKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result: Vec<f64> = values
        .iter()
        .map(|&prob| {
            mlfinance_backtesting::bet_sizing::probability_to_size::sigmoid_bet_size(
                prob,
                kwargs.num_classes as usize,
            )
        })
        .collect();
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn power_bet_size(inputs: &[Series], kwargs: PowerBetKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result: Vec<f64> = values
        .iter()
        .map(|&prob| {
            mlfinance_backtesting::bet_sizing::probability_to_size::power_bet_size(
                prob,
                kwargs.num_classes as usize,
                kwargs.exponent,
            )
        })
        .collect();
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn discrete_signal(inputs: &[Series], kwargs: DiscreteKwargs) -> PolarsResult<Series> {
    let values = extract_f64(&inputs[0])?;
    let result: Vec<f64> = values
        .iter()
        .map(|&signal| {
            mlfinance_backtesting::bet_sizing::discretization::discrete_signal(
                signal,
                kwargs.step_size,
            )
        })
        .collect();
    Ok(make_output(&inputs[0], result))
}
