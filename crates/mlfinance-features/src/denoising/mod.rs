//! Random Matrix Theory (RMT) based denoising of financial correlation matrices.
//!
//! Implements Marcenko-Pastur distribution fitting to separate signal from noise
//! eigenvalues, and provides denoising, detoning, and optimal portfolio construction.

pub mod rmt;

pub use rmt::{
    corr_to_cov, cov_to_corr, denoise_corr, denoise_cov, detone_corr, find_max_eigenvalue, fit_kde,
    marcenko_pastur_pdf, optimal_portfolio,
};
