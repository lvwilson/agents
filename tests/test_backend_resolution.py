"""Backend-target regressions: resolver, SDK stream, and subprocess CLI."""
import json
import unittest
from unittest import mock

import httpx
import openai

from agents import agents as app
from agents.agent_pool import AgentPool
from agents.backends import create_backend
from agents.backends.openai_compat_backend import OpenAICompatBackend
from agents.llm_backend import StreamHandler


CATALOG = {"qwen-3.8-27b": "cerebras", "claude-opus-4-6": "anthropic"}
LOCAL = "http://localhost:8000/v1"


class TestResolution(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(app, "_online_models", return_value=CATALOG)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cloud_pins_demoted_for_local_requests(self):
        for provider in ("cerebras", "gemini", "kimi", "deepseek", "minimax"):
            for selection in (
                {"cli_local": True, "model_name": "qwen-3.8-27b"},
                {"local_model": "qwen-3.8-27b"},
                {"model_name": "custom-local"},
                {"config_base_url": LOCAL, "config_model": "qwen-3.8-27b"},
            ):
                with self.subTest(provider=provider, selection=selection):
                    target = app.resolve_backend_target(
                        config_provider=provider, **selection
                    )
                    self.assertEqual(target[0], "openai")
                    self.assertEqual(target[2], LOCAL)

    def test_explicit_provider_wins_over_catalog(self):
        self.assertEqual(
            app.resolve_backend_target(
                model_name="qwen-3.8-27b", cli_provider="OPENAI",
                config_base_url="https://proxy.example/v1",
            ),
            ("openai", "qwen-3.8-27b", "https://proxy.example/v1"),
        )

    def test_explicit_provider_supports_unknown_hosted_model(self):
        self.assertEqual(
            app.resolve_backend_target(
                model_name="new-cloud-model", cli_provider="cerebras",
            ),
            ("cerebras", "new-cloud-model", None),
        )

    def test_hosted_report_omits_the_overridden_provider(self):
        # A concrete hosted model beats a stale ``provider:`` pin, and
        # the report must stay useful *without* naming the loser: the
        # user wants to know where the model is actually served, not to
        # be reminded of a config entry that has no bearing on the run.
        cases = (
            ({"config_provider": "anthropic", "config_model": "qwen-3.8-27b"},
             "cerebras", "anthropic"),
            ({"config_provider": "cerebras", "model_name": "claude-opus-4-6"},
             "anthropic", "cerebras"),
        )
        for kwargs, hosting, overridden in cases:
            model = kwargs.get("config_model") or kwargs["model_name"]
            with self.subTest(**kwargs), \
                    mock.patch.object(app, "safe_console_print") as console:
                resolved = app.resolve_backend_target(**kwargs)
            self.assertEqual(
                resolved[0], hosting,
                "the hosted model must still select its provider",
            )
            console.assert_called_once()
            message = console.call_args.args[0]
            self.assertIn(model, message)
            self.assertIn(hosting, message)
            self.assertNotIn(overridden, message)

    def test_proxy_retained_for_explicit_and_configured_models(self):
        url = "https://proxy.example/custom/v1"
        for model in ("qwen-3.8-27b", "custom-model"):
            for key in ("model_name", "config_model"):
                with self.subTest(model=model, key=key):
                    self.assertEqual(
                        app.resolve_backend_target(
                            config_provider="cerebras",
                            config_base_url=url, **{key: model},
                        ),
                        ("cerebras", model, url),
                    )

    def test_unpinned_proxy_defaults_to_chat_completions(self):
        for key in ("model_name", "config_model"):
            self.assertEqual(
                app.resolve_backend_target(
                    config_base_url="https://proxy.example/v1",
                    **{key: "custom-model"},
                ),
                ("openai", "custom-model", "https://proxy.example/v1"),
            )

    def test_loopback_detection(self):
        for url in (
            "http://[::1]/v1", "http://[::1]:8000/v1",
            "http://127.0.0.2/v1", "http://LOCALHOST.:8000/v1",
            "http://0.0.0.0:8000/v1",
        ):
            with self.subTest(url=url):
                self.assertTrue(app._looks_local_base_url(url))
                self.assertEqual(
                    app.resolve_backend_target(
                        config_provider="cerebras",
                        config_model="qwen-3.8-27b", config_base_url=url,
                    ),
                    ("openai", "qwen-3.8-27b", url),
                )
        for url in ("https://proxy.example/v1", "http://192.168.1.50/v1",
                    "http://[2001:db8::1]/v1", "", "http://[invalid"):
            with self.subTest(url=url):
                self.assertFalse(app._looks_local_base_url(url))

    def test_local_missing_model_fails_cleanly(self):
        for kwargs in ({"cli_local": True}, {"config_base_url": LOCAL}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                app.resolve_backend_target(config_provider="cerebras", **kwargs)

    def test_local_model_and_destination_precedence(self):
        self.assertEqual(
            app.resolve_backend_target(
                cli_local=True, model_name="explicit", local_model="env",
                config_model="config", config_provider="cerebras",
                local_host="::1", local_port=11434,
            ),
            ("openai", "explicit", "http://[::1]:11434/v1"),
        )
        self.assertEqual(
            app.resolve_backend_target(
                local_model="local", config_provider="cerebras",
                config_base_url="https://cloud-proxy.example/v1",
                local_host="192.168.1.50",
            ),
            ("openai", "local", "http://192.168.1.50:8000/v1"),
        )

    def test_explicit_local_wire_format_is_preserved(self):
        for kwargs in (
            {"cli_provider": "cerebras"},
            {"config_provider": "anthropic"},
        ):
            expected = kwargs.get("cli_provider", kwargs.get("config_provider"))
            self.assertEqual(
                app.resolve_backend_target(
                    cli_local=True, model_name="model", **kwargs,
                ),
                (expected, "model", LOCAL),
            )

    def test_local_reasoning_reaches_handler_through_real_sdk(self):
        provider, model, url = app.resolve_backend_target(
            cli_local=True, model_name="local-model", config_provider="cerebras",
        )
        handler = mock.Mock(spec=StreamHandler)
        backend = create_backend(provider, model=model, base_url=url,
                                 stream_handler=handler)
        self.assertIs(type(backend), OpenAICompatBackend)
        chunks = [
            {"id": "test", "object": "chat.completion.chunk", "created": 0,
             "model": model, "choices": [
                 {"index": 0, "delta": delta, "finish_reason": None}
             ]}
            for delta in ({"reasoning_content": "Thinking"},
                          {"content": "PONG"})
        ]
        wire = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                content=wire + "data: [DONE]\n\n",
            )

        backend._client.close()
        with openai.OpenAI(
            api_key="local", base_url=url,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            backend._client = client
            self.assertEqual(backend.generate_response("system", [
                {"role": "user", "content": [{"type": "text", "text": "ping"}]},
            ]), "PONG")
        self.assertEqual(requests[0].url.path, "/v1/chat/completions")
        handler.on_stream_reasoning_token.assert_called_once_with("Thinking")

    def test_explicit_url_overrides_config_in_environment_local_mode(self):
        self.assertEqual(
            app.resolve_backend_target(
                local_model="local-model", config_provider="cerebras",
                cli_base_url="http://192.168.1.50:11434/v1",
                config_base_url="https://stale-proxy.example/v1",
            ),
            ("openai", "local-model", "http://192.168.1.50:11434/v1"),
        )

    def test_agent_passes_resolved_target_to_backend_and_pool(self):
        config = {"system_prompt": "system", "overbudget": "stop",
                  "provider": "cerebras", "model": "qwen-3.8-27b"}
        with mock.patch.object(app, "read_configuration", return_value=config), \
             mock.patch.object(app, "load_agent_config", return_value=config), \
             mock.patch.object(app, "build_context_guard", return_value=""), \
             mock.patch.object(app, "notes_need_compact", return_value=False), \
             mock.patch.object(app, "create_backend") as backend, \
             mock.patch.object(app, "print_banner"):
            backend.return_value.context_window_size = 0
            agent = app.Agent(
                "fake.yaml", "task", session_id="routing",
                cli_local=True, model="local-model",
                base_url="http://192.168.1.50:11434/v1",
            )
        self.assertEqual(backend.call_args.args[0], "openai")
        self.assertEqual(backend.call_args.kwargs["model"], "local-model")
        self.assertEqual(backend.call_args.kwargs["base_url"],
                         "http://192.168.1.50:11434/v1")
        self.assertEqual(agent._agent_pool.provider, "openai")
        self.assertEqual(agent._agent_pool.model, "local-model")
        self.assertEqual(agent._agent_pool.base_url,
                         "http://192.168.1.50:11434/v1")


class TestDelegation(unittest.TestCase):
    def test_exact_target_round_trips_through_child_cli(self):
        for provider, model, url in (
            ("openai", "qwen-3.8-27b", "http://192.168.1.50:11434/v1"),
            ("openai", "custom", "https://proxy.example/prefix/v1"),
            ("anthropic", "custom", "http://[::1]:9000/custom"),
            ("cerebras", "new-hosted-model", None),
        ):
            with self.subTest(provider=provider, url=url):
                pool = AgentPool()
                pool.model, pool.provider, pool.base_url = model, provider, url
                pool.create("worker", "Test role")
                with mock.patch("agents.agent_pool.subprocess.run") as run:
                    run.return_value.stdout = "ok"
                    pool.run("worker", "task")
                cmd = run.call_args.args[0]
                self.assertNotIn("-p", cmd)
                with mock.patch("sys.argv", ["agent", *cmd[3:]]), \
                     mock.patch.object(app, "run_agent",
                                       return_value=("ok", True, "test")) as child, \
                     mock.patch.object(app, "report_home_config"), \
                     mock.patch.object(app.signal, "signal"), \
                     mock.patch.object(app.sys.stdin, "isatty", return_value=True), \
                     mock.patch.object(app, "safe_console_print"), \
                     mock.patch.object(app, "print_completion_result"):
                    app.main()
                kwargs = child.call_args.kwargs
                self.assertEqual(kwargs["provider"], provider)
                self.assertEqual(kwargs["model"], model)
                self.assertEqual(kwargs["base_url"], url or "")
                self.assertFalse(kwargs["planning_mode"])
                self.assertEqual(
                    app.resolve_backend_target(
                        model_name=kwargs["model"],
                        cli_provider=kwargs["provider"],
                        cli_base_url=kwargs["base_url"],
                        config_provider="cerebras",
                        config_base_url="https://stale-proxy.example/v1",
                    ),
                    (provider, model, url),
                )


if __name__ == "__main__":
    unittest.main()
