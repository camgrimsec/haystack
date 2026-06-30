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

    def test_key_ordering_does_not_force_reprompt(self, addition_tool):
        # The same parameters expressed with different key insertion orders represent the same call and must not
        # cause a second prompt.
        policy = AskOncePolicy()
        policy.update_after_confirmation(
            addition_tool.name,
            addition_tool.description,
            {"x": 1, "y": 2},
            ConfirmationUIResult(action="confirm", feedback=None),
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, {"y": 2, "x": 1}) is False

    def test_rejected_calls_do_not_silence_future_prompts(self, addition_tool):
        # Rejecting a tool call must not cause future calls with the same parameters to skip the prompt.
        policy = AskOncePolicy()
        params = {"x": 1, "y": 2}
        policy.update_after_confirmation(
            addition_tool.name, addition_tool.description, params, ConfirmationUIResult(action="reject", feedback="no")
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is True

    def test_concurrent_updates_are_safe(self, addition_tool):
        # The policy is documented as safe to share across threads (e.g. a FastAPI server hosting an agent).
        # Run many concurrent updates and assert no confirmations are dropped.
        import threading as _threading

        policy = AskOncePolicy()
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
