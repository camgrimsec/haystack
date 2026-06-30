# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.human_in_the_loop import AlwaysAskPolicy, AskOncePolicy, ConfirmationUIResult, NeverAskPolicy
from haystack.tools import Tool, create_tool_from_function


def addition(x: int, y: int) -> int:
    return x + y


@pytest.fixture
def addition_tool() -> Tool:
    return create_tool_from_function(function=addition, name="Addition tool", description="Adds two integers together.")


class TestAlwaysAskPolicy:
    def test_should_ask_always_true(self, addition_tool):
        policy = AlwaysAskPolicy()
        assert policy.should_ask(addition_tool.name, addition_tool.description, {"x": 1, "y": 2}) is True

    def test_to_dict(self):
        policy = AlwaysAskPolicy()
        policy_dict = policy.to_dict()
        assert policy_dict["type"] == "haystack.human_in_the_loop.policies.AlwaysAskPolicy"
        assert policy_dict["init_parameters"] == {}

    def test_from_dict(self):
        policy_dict = {"type": "haystack.human_in_the_loop.policies.AlwaysAskPolicy", "init_parameters": {}}
        policy = AlwaysAskPolicy.from_dict(policy_dict)
        assert isinstance(policy, AlwaysAskPolicy)


class TestAskOncePolicy:
    def test_should_ask_first_time_true(self, addition_tool):
        policy = AskOncePolicy()
        assert policy.should_ask(addition_tool.name, addition_tool.description, {"x": 1, "y": 2}) is True

    def test_should_ask_second_time_false(self, addition_tool):
        policy = AskOncePolicy()
        params = {"x": 1, "y": 2}
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is True
        # Simulate the update after confirmation that occurs in HumanInTheLoopStrategy
        policy.update_after_confirmation(
            addition_tool.name, addition_tool.description, params, ConfirmationUIResult(action="confirm", feedback=None)
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is False

    def test_should_ask_different_params_true(self, addition_tool):
        policy = AskOncePolicy()
        params1 = {"x": 1, "y": 2}
        params2 = {"x": 3, "y": 4}
        assert policy.should_ask(addition_tool.name, addition_tool.description, params1) is True
        # Simulate the update after confirmation that occurs in HumanInTheLoopStrategy
        policy.update_after_confirmation(
            addition_tool.name,
            addition_tool.description,
            params1,
            ConfirmationUIResult(action="confirm", feedback=None),
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params2) is True

    def test_to_dict(self):
        policy = AskOncePolicy()
        policy_dict = policy.to_dict()
        assert policy_dict["type"] == "haystack.human_in_the_loop.policies.AskOncePolicy"
        assert policy_dict["init_parameters"] == {}

    def test_from_dict(self):
        policy_dict = {"type": "haystack.human_in_the_loop.policies.AskOncePolicy", "init_parameters": {}}
        policy = AskOncePolicy.from_dict(policy_dict)
        assert isinstance(policy, AskOncePolicy)

    def test_remembers_multiple_confirmed_param_sets(self, addition_tool):
        # A previously confirmed parameter set must remain confirmed even after a different parameter set is
        # confirmed for the same tool. Without this guarantee, an agent that re-issues a previously confirmed call
        # after running other tool calls would re-prompt the user.
        policy = AskOncePolicy()
        params1 = {"x": 1, "y": 2}
        params2 = {"x": 3, "y": 4}

        policy.update_after_confirmation(
            addition_tool.name,
            addition_tool.description,
            params1,
            ConfirmationUIResult(action="confirm", feedback=None),
        )
        policy.update_after_confirmation(
            addition_tool.name,
            addition_tool.description,
            params2,
            ConfirmationUIResult(action="confirm", feedback=None),
        )

        assert policy.should_ask(addition_tool.name, addition_tool.description, params1) is False
        assert policy.should_ask(addition_tool.name, addition_tool.description, params2) is False

    def test_rejected_calls_do_not_silence_future_prompts(self, addition_tool):
        # Rejecting a tool call must not cause future calls with the same parameters to skip the prompt.
        policy = AskOncePolicy()
        params = {"x": 1, "y": 2}
        policy.update_after_confirmation(
            addition_tool.name, addition_tool.description, params, ConfirmationUIResult(action="reject", feedback="no")
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is True

    def test_modified_calls_do_not_silence_future_prompts(self, addition_tool):
        # A "modify" outcome means the user wanted to intervene; the next call with the same parameters should
        # still prompt.
        policy = AskOncePolicy()
        params = {"x": 1, "y": 2}
        policy.update_after_confirmation(
            addition_tool.name,
            addition_tool.description,
            params,
            ConfirmationUIResult(action="modify", feedback="change y to 5"),
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is True

    @pytest.mark.parametrize("non_json_native_params", [{"x": b"bytes"}, {"x": {1, 2, 3}}, {"x": object()}])
    def test_non_json_native_params_re_prompt(self, addition_tool, non_json_native_params):
        # Params containing values that json.dumps cannot encode natively (bytes, sets, custom objects, ...) must
        # cause a re-prompt rather than silently collapsing onto the same cache key as another distinct call. This
        # is the conservative choice: a spurious extra prompt is far better than wrongly suppressing one.
        policy = AskOncePolicy()
        # Even after a "confirm" update with the same non-JSON-native params, should_ask must still return True.
        policy.update_after_confirmation(
            addition_tool.name,
            addition_tool.description,
            non_json_native_params,
            ConfirmationUIResult(action="confirm", feedback=None),
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, non_json_native_params) is True

    def test_clear_resets_all_state(self, addition_tool):
        # ``clear()`` with no argument drops every cached confirmation, restoring the policy to its initial state.
        # This is the intended hook for a session / user / tenant boundary.
        policy = AskOncePolicy()
        params = {"x": 1, "y": 2}
        policy.update_after_confirmation(
            addition_tool.name, addition_tool.description, params, ConfirmationUIResult(action="confirm", feedback=None)
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is False
        policy.clear()
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is True

    def test_clear_specific_tool(self, addition_tool):
        # ``clear(tool_name=...)`` drops cached confirmations for one tool only, leaving others intact.
        policy = AskOncePolicy()
        params = {"x": 1, "y": 2}
        other_tool = "other_tool"
        policy.update_after_confirmation(
            addition_tool.name, addition_tool.description, params, ConfirmationUIResult(action="confirm", feedback=None)
        )
        policy.update_after_confirmation(
            other_tool, "other", params, ConfirmationUIResult(action="confirm", feedback=None)
        )

        policy.clear(tool_name=addition_tool.name)

        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is True
        assert policy.should_ask(other_tool, "other", params) is False

    def test_max_entries_per_tool_evicts_fifo(self, addition_tool):
        # With a small cap, confirming more distinct parameter sets than the cap allows must evict the oldest
        # entries first so memory stays bounded. The most-recently-confirmed entries are retained.
        policy = AskOncePolicy(max_entries_per_tool=3)
        param_sets = [{"x": i, "y": i + 1} for i in range(5)]
        for params in param_sets:
            policy.update_after_confirmation(
                addition_tool.name,
                addition_tool.description,
                params,
                ConfirmationUIResult(action="confirm", feedback=None),
            )

        # The two oldest entries should have been evicted and should now re-prompt.
        assert policy.should_ask(addition_tool.name, addition_tool.description, param_sets[0]) is True
        assert policy.should_ask(addition_tool.name, addition_tool.description, param_sets[1]) is True
        # The three most-recent entries should still be cached.
        for params in param_sets[2:]:
            assert policy.should_ask(addition_tool.name, addition_tool.description, params) is False

    def test_max_entries_per_tool_invalid_value(self):
        with pytest.raises(ValueError):
            AskOncePolicy(max_entries_per_tool=0)

    def test_concurrent_updates_do_not_drop_writes(self, addition_tool):
        # The in-process lock protects cache mutation: many threads updating the cache concurrently must not lose
        # writes or corrupt the underlying OrderedDict. This is a narrow guarantee; see the policy docstring for
        # what the lock does NOT cover (notably the ask -> UI -> update window and multi-worker deployments).
        import threading as _threading

        policy = AskOncePolicy(max_entries_per_tool=500)
        all_params = [{"x": i, "y": i + 1} for i in range(200)]

        def worker(params):
            policy.update_after_confirmation(
                addition_tool.name,
                addition_tool.description,
                params,
                ConfirmationUIResult(action="confirm", feedback=None),
            )

        threads = [_threading.Thread(target=worker, args=(p,)) for p in all_params]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for params in all_params:
            assert policy.should_ask(addition_tool.name, addition_tool.description, params) is False


class TestNeverAskPolicy:
    def test_should_ask_always_false(self, addition_tool):
        policy = NeverAskPolicy()
        assert policy.should_ask(addition_tool.name, addition_tool.description, {"x": 1, "y": 2}) is False

    def test_to_dict(self):
        policy = NeverAskPolicy()
        policy_dict = policy.to_dict()
        assert policy_dict["type"] == "haystack.human_in_the_loop.policies.NeverAskPolicy"
        assert policy_dict["init_parameters"] == {}

    def test_from_dict(self):
        policy_dict = {"type": "haystack.human_in_the_loop.policies.NeverAskPolicy", "init_parameters": {}}
        policy = NeverAskPolicy.from_dict(policy_dict)
        assert isinstance(policy, NeverAskPolicy)
