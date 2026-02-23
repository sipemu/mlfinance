use serde::Deserialize;

// -- Core --

#[derive(Deserialize)]
pub struct EwmaKwargs {
    pub span: u64,
}

// -- Sampling --

#[derive(Deserialize)]
pub struct FracDiffKwargs {
    pub d: f64,
    pub threshold: f64,
}

#[derive(Deserialize)]
pub struct FindMinDKwargs {
    pub max_d: f64,
    pub step_size: f64,
    pub threshold: f64,
}

// -- Labeling --

#[derive(Deserialize)]
pub struct VolKwargs {
    pub span: u64,
}

#[derive(Deserialize)]
pub struct TrendScanKwargs {
    pub max_window: u64,
}

// -- Features --

#[derive(Deserialize)]
pub struct AdfKwargs {
    pub max_lags: u64,
}

#[derive(Deserialize)]
pub struct SadfKwargs {
    pub min_window: u64,
    pub max_lags: u64,
}

#[derive(Deserialize)]
pub struct QuantileKwargs {
    pub n_bins: u64,
}

#[derive(Deserialize)]
pub struct SigmaKwargs {
    pub n_bands: u64,
}

#[derive(Deserialize)]
pub struct EntropyKwargs {
    pub window: u64,
}

// -- Backtesting --

#[derive(Deserialize)]
pub struct SharpeKwargs {
    pub risk_free_rate: f64,
    pub periods_per_year: f64,
}

#[derive(Deserialize)]
pub struct BetSizeKwargs {
    pub num_classes: u64,
}

#[derive(Deserialize)]
pub struct PowerBetKwargs {
    pub num_classes: u64,
    pub exponent: f64,
}

#[derive(Deserialize)]
pub struct DiscreteKwargs {
    pub step_size: f64,
}

// -- Multi-column --

#[derive(Deserialize)]
pub struct WindowKwargs {
    pub window: u64,
}

#[derive(Deserialize)]
pub struct VpinKwargs {
    pub bucket_size: f64,
    pub n_buckets: u64,
}
