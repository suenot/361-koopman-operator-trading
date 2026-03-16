//! Koopman operator methods for time series analysis
//!
//! This module provides implementations of:
//! - DMD (Dynamic Mode Decomposition)
//! - EDMD (Extended DMD) with various dictionary functions
//! - Prediction utilities

mod dmd;
mod edmd;
mod prediction;

pub use dmd::DMD;
pub use edmd::{Dictionary, EDMD, PolynomialDictionary, RBFDictionary};
pub use prediction::{delay_embed, optimal_rank};
