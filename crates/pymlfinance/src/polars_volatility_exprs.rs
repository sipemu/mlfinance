use chrono::Utc;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::prelude::*;

use mlfinance_core::types::OhlcvBar;

use crate::polars_convert::*;
use crate::polars_kwargs::WindowKwargs;

fn map_err(e: impl std::fmt::Display) -> PolarsError {
    PolarsError::ComputeError(format!("{e}").into())
}

fn make_bars_hl(highs: &[f64], lows: &[f64]) -> Vec<OhlcvBar> {
    let dummy_ts = Utc::now();
    highs
        .iter()
        .zip(lows)
        .map(|(&h, &l)| OhlcvBar {
            timestamp: dummy_ts,
            open: h,
            high: h,
            low: l,
            close: l,
            volume: 0.0,
            vwap: 0.0,
        })
        .collect()
}

fn make_bars_ohlc(opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<OhlcvBar> {
    let dummy_ts = Utc::now();
    opens
        .iter()
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .map(|(((&o, &h), &l), &c)| OhlcvBar {
            timestamp: dummy_ts,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: 0.0,
            vwap: 0.0,
        })
        .collect()
}

#[polars_expr(output_type=Float64)]
fn parkinson_volatility(inputs: &[Series], kwargs: WindowKwargs) -> PolarsResult<Series> {
    let highs = extract_f64(&inputs[0])?;
    let lows = extract_f64(&inputs[1])?;
    let bars = make_bars_hl(&highs, &lows);
    let result =
        mlfinance_labeling::volatility::parkinson_volatility(&bars, kwargs.window as usize)
            .map_err(map_err)?;
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn garman_klass_volatility(inputs: &[Series], kwargs: WindowKwargs) -> PolarsResult<Series> {
    let opens = extract_f64(&inputs[0])?;
    let highs = extract_f64(&inputs[1])?;
    let lows = extract_f64(&inputs[2])?;
    let closes = extract_f64(&inputs[3])?;
    let bars = make_bars_ohlc(&opens, &highs, &lows, &closes);
    let result =
        mlfinance_labeling::volatility::garman_klass_volatility(&bars, kwargs.window as usize)
            .map_err(map_err)?;
    Ok(make_output(&inputs[0], result))
}

#[polars_expr(output_type=Float64)]
fn yang_zhang_volatility(inputs: &[Series], kwargs: WindowKwargs) -> PolarsResult<Series> {
    let opens = extract_f64(&inputs[0])?;
    let highs = extract_f64(&inputs[1])?;
    let lows = extract_f64(&inputs[2])?;
    let closes = extract_f64(&inputs[3])?;
    let bars = make_bars_ohlc(&opens, &highs, &lows, &closes);
    let result =
        mlfinance_labeling::volatility::yang_zhang_volatility(&bars, kwargs.window as usize)
            .map_err(map_err)?;
    Ok(make_output(&inputs[0], result))
}
