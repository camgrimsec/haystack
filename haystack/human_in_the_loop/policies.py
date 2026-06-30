# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from typing import Any

from haystack.human_in_the_loop import ConfirmationUIResult
from haystack.human_in_the_loop.types import ConfirmationPolicy


class AlwaysAskPolicy(ConfirmationPolicy):
    """Always ask for confirmation."""

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:  # noqa: ARG002
        """
        Always ask for confirmation before executing the tool.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns True, indicating confirmation is needed.
        """
        return True


class NeverAskPolicy(ConfirmationPolicy):
    """Never ask for confirmation."""

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:  # noqa: ARG002
        """
        Never ask for confirmation, always proceed with tool execution.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns False, indicating no confirmation is needed.
        """
        return False


class AskOncePolicy(ConfirmationPolicy):
    """Ask only once per tool with specific parameters."""

    def __init__(self) -> None:
        """Creates an instance of AskOncePolicy."""
        # Each tool_name maps to a set of canonical-JSON-encoded parameter sets that the user has already confirmed.
        # Using a set (instead of a single dict overwritten per call) is required so that consecutive calls with
        # different parameter sets do not erase each other's confirmation state.
        self._confirmed_params: dict[str, set[str]] = {}
        # The policy can be shared across requests / threads when the surrounding agent is hosted in a server
        # (FastAPI, etc.). Without a lock, two threads can interleave should_ask + update_after_confirmation and
        # either double-prompt the user or skip a prompt the user should have seen.
        self._lock = threading.Lock()

    @staticmethod
    def _canonicalize(tool_params: dict[str, Any]) -> str:
        """Return a deterministic string key for a tool-parameter dict so equality is stable."""
        try:
            return json.dumps(tool_params, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to repr for objects that are not JSON-serialisable. repr() is stable for primitive
            # containers and deterministic enough for cache-key purposes, even if it is less canonical.
            return repr(tool_params)

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:  # noqa: ARG002
        """
        Ask for confirmation only once per tool with specific parameters.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns: True if confirmation is needed, False if the user has already confirmed this exact parameter set
            for this tool earlier in the session.
        """
        key = self._canonicalize(tool_params)
        with self._lock:
            confirmed = self._confirmed_params.get(tool_name)
            return not (confirmed is not None and key in confirmed)

    def update_after_confirmation(
        self,
        tool_name: str,
        tool_description: str,  # noqa: ARG002
        tool_params: dict[str, Any],
        confirmation_result: ConfirmationUIResult,
    ) -> None:
        """
        Store the tool and parameters if the action was "confirm" to avoid asking again.

        This method updates the internal state to remember that the user has already confirmed the execution of the
        tool with the given parameters. Confirmations are remembered per (tool_name, parameter set), so confirming
        a tool with one parameter set does not implicitly confirm or forget any other parameter set.

        :param tool_name: The name of the tool that was executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters that were passed to the tool.
        :param confirmation_result: The result from the confirmation UI.
        """
        if confirmation_result.action != "confirm":
            return
        key = self._canonicalize(tool_params)
        with self._lock:
            self._confirmed_params.setdefault(tool_name, set()).add(key)
