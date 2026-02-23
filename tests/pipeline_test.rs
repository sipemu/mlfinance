//! End-to-end pipeline test: ticks → bars → CUSUM → volatility → triple barrier
//! → labels → indicator matrix → sample weights → FFD → purged k-fold → bet sizing → Sharpe

use mlfinance::backtesting::bet_sizing::probability_to_size::sigmoid_bet_size;
use mlfinance::backtesting::statistics::sharpe::sharpe_ratio;
use mlfinance::core::traits::BarAggregator;
use mlfinance::core::types::TickData;
use mlfinance::data::bars::tick_bars::TickBarAggregator;
use mlfinance::data::sampling::cusum_filter::cusum_filter;
use mlfinance::labeling::barriers::TripleBarrierConfig;
use mlfinance::labeling::events::get_events;
use mlfinance::labeling::labels::get_bins;
use mlfinance::modeling::cross_validation::purged_kfold::PurgedKFold;
use mlfinance::sampling::concurrency::indicator_matrix::get_indicator_matrix;
use mlfinance::sampling::fracdiff::ffd::frac_diff_ffd;

fn generate_synthetic_ticks(n: usize) -> Vec<TickData> {
    let mut price = 100.0;
    (0..n)
        .map(|i| {
            // Synthetic price with trend + mean reversion + noise
            price += (i as f64 * 0.03).sin() * 0.5 + ((i * 7) as f64 * 0.1).sin() * 0.2;
            let volume = 100.0 + (i as f64 * 0.07).cos() * 50.0;
            TickData {
                timestamp: chrono::DateTime::from_timestamp(
                    1_700_000_000 + i as i64,
                    0,
                )
                .unwrap(),
                price: price.max(1.0), // ensure positive price
                volume: volume.abs(),
            }
        })
        .collect()
}

#[test]
fn test_full_book_pipeline() {
    // Stage 1: Generate synthetic tick data
    let ticks = generate_synthetic_ticks(5000);
    assert_eq!(ticks.len(), 5000);

    // Stage 2: Aggregate into tick bars
    let mut agg = TickBarAggregator::new(20);
    let bars = agg.process_ticks(&ticks);
    assert!(bars.len() > 100, "Should produce >100 bars from 5000 ticks");

    // Stage 3: Extract close prices
    let prices: Vec<f64> = bars.iter().map(|b| b.close).collect();
    assert_eq!(prices.len(), bars.len());

    // Stage 4: CUSUM filter to find events
    let cusum_events = cusum_filter(&prices, 1.0);
    assert!(
        !cusum_events.is_empty(),
        "CUSUM should detect at least some events"
    );
    for &idx in &cusum_events {
        assert!(idx < prices.len());
    }

    // Stage 5: Set up daily volatilities (use simple estimate)
    let daily_vols = vec![0.02; prices.len()]; // 2% daily vol estimate

    // Stage 6: Triple barrier labeling
    let config = TripleBarrierConfig {
        upper_barrier: Some(2.0),
        lower_barrier: Some(2.0),
        max_holding_period: Some(20),
    };
    let events = get_events(&prices, &cusum_events, &config, &daily_vols);
    assert!(
        !events.is_empty(),
        "Should produce some labeled events"
    );

    // Stage 7: Get labels
    let labels = get_bins(&events);
    assert_eq!(labels.len(), events.len());
    for &l in &labels {
        assert!(l == -1 || l == 0 || l == 1);
    }

    // Stage 8: Build indicator matrix from event spans
    let event_spans: Vec<(usize, usize)> = events
        .iter()
        .map(|e| (e.entry_idx, e.exit_idx))
        .collect();
    let ind_matrix = get_indicator_matrix(&event_spans, prices.len());
    assert_eq!(ind_matrix.ncols(), events.len());

    // Stage 9: FFD on prices
    let ffd_prices = frac_diff_ffd(&prices, 0.5, 1e-4);
    assert_eq!(ffd_prices.len(), prices.len());
    // First few values should be NaN, rest should be finite
    let non_nan_count = ffd_prices.iter().filter(|v| !v.is_nan()).count();
    assert!(non_nan_count > prices.len() / 2);

    // Stage 10: Purged K-Fold
    if events.len() >= 6 {
        let n_splits = 3.min(events.len() / 2);
        let kfold = PurgedKFold::new(n_splits, 0.02);
        let folds = kfold.split(&event_spans, events.len());
        assert_eq!(folds.len(), n_splits);

        // Every sample in exactly one test set
        let mut test_counts = vec![0usize; events.len()];
        for fold in &folds {
            for &idx in &fold.test {
                test_counts[idx] += 1;
            }
        }
        for &count in &test_counts {
            assert_eq!(count, 1);
        }
    }

    // Stage 11: Bet sizing from mock probabilities
    let bet_sizes: Vec<f64> = (0..events.len())
        .map(|i| sigmoid_bet_size(0.3 + 0.4 * (i as f64 / events.len() as f64), 2))
        .collect();
    for &s in &bet_sizes {
        assert!(s >= -1.0 && s <= 1.0);
    }

    // Stage 12: Compute Sharpe ratio on synthetic returns
    let mock_returns: Vec<f64> = events
        .iter()
        .map(|e| e.return_value * 0.5) // scale down returns
        .collect();
    let sr = sharpe_ratio(&mock_returns, 0.0, 252.0);
    assert!(sr.is_finite(), "Sharpe ratio should be finite");
}
