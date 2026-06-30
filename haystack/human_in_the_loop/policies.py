# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from collections import OrderedDict
from typing import Any

from haystack.human_in_the_loop import ConfirmationUIResult
from haystack.human_in_the_loop.types import ConfirmationPolicy

# Maximum number of distinct confirmed parameter sets cached per tool. When this cap is reached, the oldest entry
# is evicted FIFO so a long-running agent that calls the same tool with many distinct parameter sets cannot grow
# the cache without bound. The exact value is deliberately conservative; callers that need a different value can
# pass ``max_entries_per_tool`` to ``AskOncePolicy``.
_DEFAULT_MAX_ENTRIES_PER_TOOL = 256


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
    """
    Ask once per (tool_name, parameter set) and remember the answer for the lifetime of the instance.

    Important scoping notes (please read before sharing an instance across requests):

    * **Per session / per user.** A single ``AskOncePolicy`` instance retains every confirmed call until it is
      discarded or ``clear()`` is called. If the same instance is shared across users or tenants, one user's
      confirmation suppresses the prompt for any other user that issues the same tool call with the same
      parameters. For multi-user web/server deployments, instantiate the policy per session.
    * **The lock is in-process and protects only the cache mutation.** Two concurrent ``should_ask`` calls can both
      see "not yet confirmed", show two prompts to the user, and both update afterwards. The lock prevents lost
      writes and corrupt reads of the internal cache, not double-prompting across the ask -> UI -> update window.
    * **Multi-worker deployments.** Because the lock is in-process, a Gunicorn / Uvicorn / Celery deployment with
      multiple worker processes has one cache per worker. Confirmation state is not shared across workers.
    * **JSON-native parameters only.** The confirmation cache is keyed by a canonical JSON encoding of the
      parameter set. If a tool is invoked with non-JSON-native values (custom objects, ``bytes``, sets, ...), the
      policy errs on the side of safety and re-prompts, rather than collapsing distinct calls onto the same cache
      key.
    """

    def __init__(self, max_entries_per_tool: int = _DEFAULT_MAX_ENTRIES_PER_TOOL) -> None:
        """
        Creates an instance of AskOncePolicy.

        :param max_entries_per_tool:
            Maximum number of distinct confirmed parameter sets cached per tool. Older entries are evicted FIFO
            once this cap is reached. Defaults to 256; set to a smaller value for memory-sensitive deployments or
            a larger value if your agent legitimately invokes the same tool with thousands of distinct parameter
            sets within a single session.
        """
        if max_entries_per_tool < 1:
            raise ValueError("max_entries_per_tool must be >= 1")
        self._max_entries_per_tool = max_entries_per_tool
        # Each tool_name maps to an OrderedDict acting as an LRU-ish FIFO cache of canonical-JSON-encoded
        # parameter sets that the user has already confirmed. The mapped value is always True; only the key
        # presence and insertion order matter.
        self._confirmed_params: dict[str, OrderedDict[str, bool]] = {}
        # Protects mutation and read of ``_confirmed_params`` from being interleaved across threads in a single
        # Python process. See the class docstring for what this lock does NOT protect.
        self._lock = threading.Lock()

    @staticmethod
    def _canonicalize(tool_params: dict[str, Any]) -> str | None:
        """
        Return a deterministic string key for JSON-native params, or None for non-JSON-native input.

        A ``None`` return tells the caller to behave as if the call had never been confirmed before, which
        causes the user to be prompted again rather than silently colliding two semantically distinct calls
        onto the same cache key.
        """
        try:
            # ``allow_nan=False`` keeps the encoding tight (no "NaN" / "Infinity" tokens), and the omission of
            # ``default=`` means non-JSON-native objects raise TypeError instead of being coerced via ``str()``.
            return json.dumps(tool_params, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return None

    def should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:  # noqa: ARG002
        """
        Ask for confirmation unless the same (tool_name, parameter set) has been confirmed earlier in the session.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters to be passed to the tool.
        :returns:
            True if confirmation is needed. False only if the user has already confirmed this exact, JSON-native
            parameter set for this tool. Returns True (ask again) whenever the parameters are not JSON-native, to
            avoid suppressing a prompt for a semantically different call.
        """
        key = self._canonicalize(tool_params)
        if key is None:
            return True
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
        Store the (tool_name, parameter set) pair if the user confirmed the call.

        Confirmations are remembered per (tool_name, parameter set); confirming a tool with one parameter set does
        not implicitly confirm any other parameter set. ``"reject"`` and ``"modify"`` results are intentionally not
        cached, so the user remains in the loop the next time the agent attempts an equivalent call.

        :param tool_name: The name of the tool that was executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters that were passed to the tool.
        :param confirmation_result: The result from the confirmation UI.
        """
        if confirmation_result.action != "confirm":
            return
        key = self._canonicalize(tool_params)
        if key is None:
            return
        with self._lock:
            bucket = self._confirmed_params.setdefault(tool_name, OrderedDict())
            if key in bucket:
                # Touch insertion order so frequently re-used entries are evicted last.
                bucket.move_to_end(key)
            else:
                bucket[key] = True
                while len(bucket) > self._max_entries_per_tool:
                    bucket.popitem(last=False)

    def clear(self, tool_name: str | None = None) -> None:
        """
        Discard cached confirmations.

        Use this at a session / user / tenant boundary so that one user's confirmations cannot suppress a prompt
        for another user. Without an argument, all cached confirmations are dropped. With a ``tool_name`` only the
        cached confirmations for that tool are dropped.

        :param tool_name: If provided, only clear cached confirmations for this tool; otherwise clear everything.
        """
        with self._lock:
            if tool_name is None:
                self._confirmed_params.clear()
            else:
                self._confirmed_params.pop(tool_name, None)
