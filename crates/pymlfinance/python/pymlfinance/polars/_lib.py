"""Tier 2: Multi-column standalone functions.

These take multiple column expressions as arguments and are used as
standalone functions rather than through the .ml namespace.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from pymlfinance.polars import LIB

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


# ---------------------------------------------------------------------------
# Microstructure (multi-column)
# ---------------------------------------------------------------------------

def amihud_lambda(returns: IntoExpr, dollar_volumes: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[returns, dollar_volumes],
        function_name="amihud_lambda",
        returns_scalar=True,
    )


def amihud_lambda_rolling(
    returns: IntoExpr, dollar_volumes: IntoExpr, window: int = 20
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[returns, dollar_volumes],
        function_name="amihud_lambda_rolling",
        is_elementwise=True,
        kwargs={"window": window},
    )


def kyle_lambda(returns: IntoExpr, signed_volume: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[returns, signed_volume],
        function_name="kyle_lambda",
        returns_scalar=True,
    )


def roll_spread_rolling(prices: IntoExpr, window: int = 20) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[prices],
        function_name="roll_spread_rolling",
        is_elementwise=True,
        kwargs={"window": window},
    )


def corwin_schultz_spread(highs: IntoExpr, lows: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[highs, lows],
        function_name="corwin_schultz_spread",
        is_elementwise=True,
    )


def vpin(
    volumes: IntoExpr, prices: IntoExpr, bucket_size: float = 1000.0, n_buckets: int = 10
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[volumes, prices],
        function_name="vpin",
        kwargs={"bucket_size": bucket_size, "n_buckets": n_buckets},
    )


# ---------------------------------------------------------------------------
# Volatility estimators (multi-column)
# ---------------------------------------------------------------------------

def parkinson_volatility(highs: IntoExpr, lows: IntoExpr, window: int = 20) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[highs, lows],
        function_name="parkinson_volatility",
        is_elementwise=True,
        kwargs={"window": window},
    )


def garman_klass_volatility(
    opens: IntoExpr, highs: IntoExpr, lows: IntoExpr, closes: IntoExpr, window: int = 20
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[opens, highs, lows, closes],
        function_name="garman_klass_volatility",
        is_elementwise=True,
        kwargs={"window": window},
    )


def yang_zhang_volatility(
    opens: IntoExpr, highs: IntoExpr, lows: IntoExpr, closes: IntoExpr, window: int = 20
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[opens, highs, lows, closes],
        function_name="yang_zhang_volatility",
        is_elementwise=True,
        kwargs={"window": window},
    )
