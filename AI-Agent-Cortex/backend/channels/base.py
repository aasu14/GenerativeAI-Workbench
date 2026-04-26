"""Base class for external messaging channels."""
from abc import ABC, abstractmethod


class BaseChannel(ABC):
    """Abstract base for messaging channel integrations."""

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a message to the external channel."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start listening for incoming messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel listener."""
        pass

    @abstractmethod
    async def handle_incoming(self, message: dict) -> str:
        """Handle an incoming message and return a response."""
        pass
