use pyo3_polars::export::polars_core::prelude::*;

/// Extract a `Vec<f64>` from a Series, mapping nulls to `f64::NAN`.
pub fn extract_f64(s: &Series) -> PolarsResult<Vec<f64>> {
    let ca = s.cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    Ok(ca.iter().map(|opt| opt.unwrap_or(f64::NAN)).collect())
}

/// Build a Float64 output Series with the same name as the input.
pub fn make_output(input: &Series, values: Vec<f64>) -> Series {
    Float64Chunked::from_vec(input.name().clone(), values).into_series()
}

/// Build an Int32 output Series with the same name as the input.
pub fn make_output_i32(input: &Series, values: Vec<i32>) -> Series {
    Int32Chunked::from_vec(input.name().clone(), values).into_series()
}

/// Build a scalar Float64 Series (single element).
pub fn make_scalar(input: &Series, value: f64) -> Series {
    Float64Chunked::from_vec(input.name().clone(), vec![value]).into_series()
}

/// Pad a vector with NaN at the front so it reaches `total_len`.
pub fn pad_front(values: Vec<f64>, total_len: usize) -> Vec<f64> {
    if values.len() >= total_len {
        return values;
    }
    let pad = total_len - values.len();
    let mut result = vec![f64::NAN; pad];
    result.extend(values);
    result
}
