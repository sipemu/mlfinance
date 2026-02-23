from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

# The compiled _native.so lives in the parent (pymlfinance/) directory
LIB = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Tier 1: Single-column expression namespace  .ml
# ---------------------------------------------------------------------------

@pl.api.register_expr_namespace("ml")
class MlFinanceExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # -- Core --

    def ewma(self, span: int) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="ewma",
            is_elementwise=True,
            kwargs={"span": span},
        )

    def ewma_std(self, span: int) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="ewma_std",
            is_elementwise=True,
            kwargs={"span": span},
        )

    def cumsum(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="cumsum",
            is_elementwise=True,
        )

    def log_returns(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="log_returns",
            is_elementwise=True,
        )

    def simple_returns(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="simple_returns",
            is_elementwise=True,
        )

    # -- Sampling --

    def frac_diff_ffd(self, d: float, threshold: float = 1e-4) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="frac_diff_ffd",
            is_elementwise=True,
            kwargs={"d": d, "threshold": threshold},
        )

    def frac_diff_expanding(self, d: float, threshold: float = 1e-4) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="frac_diff_expanding",
            is_elementwise=True,
            kwargs={"d": d, "threshold": threshold},
        )

    def find_min_d(
        self, max_d: float = 1.0, step_size: float = 0.1, threshold: float = 1e-4
    ) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="find_min_d",
            returns_scalar=True,
            kwargs={"max_d": max_d, "step_size": step_size, "threshold": threshold},
        )

    # -- Labeling --

    def daily_volatility(self, span: int = 100) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="daily_volatility",
            is_elementwise=True,
            kwargs={"span": span},
        )

    def trend_scanning_label_series(self, max_window: int = 20) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="trend_scanning_label_series",
            is_elementwise=True,
            kwargs={"max_window": max_window},
        )

    # -- Features: Structural Breaks --

    def adf_test(self, max_lags: int = 1) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="adf_test",
            returns_scalar=True,
            kwargs={"max_lags": max_lags},
        )

    def sadf(self, min_window: int = 20, max_lags: int = 1) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="sadf",
            is_elementwise=True,
            kwargs={"min_window": min_window, "max_lags": max_lags},
        )

    # -- Features: Encoding --

    def binary_encode(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="binary_encode",
            is_elementwise=True,
        )

    def quantile_encode(self, n_bins: int = 10) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="quantile_encode",
            is_elementwise=True,
            kwargs={"n_bins": n_bins},
        )

    def sigma_encode(self, n_bands: int = 3) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="sigma_encode",
            is_elementwise=True,
            kwargs={"n_bands": n_bands},
        )

    # -- Features: Entropy --

    def lempel_ziv_complexity(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="lempel_ziv_complexity",
            returns_scalar=True,
        )

    def kontoyiannis_entropy(self, window: int = 10) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="kontoyiannis_entropy",
            returns_scalar=True,
            kwargs={"window": window},
        )

    def shannon_entropy(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="shannon_entropy",
            returns_scalar=True,
        )

    def plugin_entropy(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="plugin_entropy",
            returns_scalar=True,
        )

    # -- Features: Microstructure (single-column) --

    def tick_rule_classify(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="tick_rule_classify",
            is_elementwise=True,
        )

    # -- Backtesting: Statistics --

    def sharpe_ratio(
        self, risk_free_rate: float = 0.0, periods_per_year: float = 252.0
    ) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="sharpe_ratio",
            returns_scalar=True,
            kwargs={
                "risk_free_rate": risk_free_rate,
                "periods_per_year": periods_per_year,
            },
        )

    def hit_ratio(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="hit_ratio",
            returns_scalar=True,
        )

    def hhi(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="hhi",
            returns_scalar=True,
        )

    def compute_drawdowns(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="compute_drawdowns",
            is_elementwise=True,
        )

    # -- Backtesting: Bet Sizing --

    def sigmoid_bet_size(self, num_classes: int = 2) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="sigmoid_bet_size",
            is_elementwise=True,
            kwargs={"num_classes": num_classes},
        )

    def power_bet_size(self, num_classes: int = 2, exponent: float = 1.0) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="power_bet_size",
            is_elementwise=True,
            kwargs={"num_classes": num_classes, "exponent": exponent},
        )

    def discrete_signal(self, step_size: float = 0.1) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[self._expr],
            function_name="discrete_signal",
            is_elementwise=True,
            kwargs={"step_size": step_size},
        )
