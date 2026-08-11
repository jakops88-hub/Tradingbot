"""Broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trading_bot.data.models import Fill, Order


class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order, price: float) -> Fill:
        """Submit an order and return the resulting fill."""
