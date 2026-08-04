"""Compatibility import for the lane-aware event bus."""

from .event_bus import Event, EventBus, EventPriority, PriorityEventBus, event_priority

__all__ = ["Event", "EventBus", "EventPriority", "PriorityEventBus", "event_priority"]
