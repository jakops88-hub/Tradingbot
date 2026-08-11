"""Broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from trading_bot.data.models import Order, Trade


class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order, market_price: Decimal) -> Trade:
        """Submit an order and return the resulting trade record."""
