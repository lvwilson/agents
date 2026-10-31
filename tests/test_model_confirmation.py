"""Consent is required before runtime routing silently falls back locally."""
import io
import unittest
from unittest import mock

from agents import agents as app, ui
from agents.cli import commit_cli


LOCAL = "http://localhost:8000/v1"
UNKNOWN = "gpt-6.1-astra"


class TestFallbackGuard(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(app, "_online_models",
                                    return_value={"known": "openai"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_implicit_fallback_requires_consent(self):
        for selection in ({"model_name": UNKNOWN}, {"config_model": UNKNOWN},
                          {"model_name": UNKNOWN, "config_provider": "cerebras"}):
            for accepted in (False, True):
                with self.subTest(selection=selection, accepted=accepted):
                    confirm = mock.Mock(return_value=accepted)
                    if accepted:
                        self.assertEqual(
                            app.resolve_backend_target(
                                **selection, confirm_local_fallback=confirm),
                            ("openai", UNKNOWN, LOCAL))
                    else:
                        with self.assertRaisesRegex(ValueError, "fallback cancelled"):
                            app.resolve_backend_target(
                                **selection, confirm_local_fallback=confirm)
                    confirm.assert_called_once_with(UNKNOWN, LOCAL)

    def test_saved_local_default_and_inherited_endpoint_require_consent(self):
        for selection in (
            {"config_model": UNKNOWN, "config_provider": "openai",
             "config_base_url": LOCAL},
            {"model_name": UNKNOWN, "config_base_url": LOCAL},
            {"config_model": UNKNOWN, "config_base_url": "http://[::1]:9000/v1"},
        ):
            with self.subTest(selection=selection):
                confirm = mock.Mock(return_value=False)
                with self.assertRaisesRegex(ValueError, "fallback cancelled"):
                    app.resolve_backend_target(**selection, confirm_local_fallback=confirm)
                confirm.assert_called_once_with(UNKNOWN, selection["config_base_url"])

    def test_explicit_local_intent_overrides_inherited_endpoint_warning(self):
        for selection in (
            {"cli_local": True, "model_name": UNKNOWN},
            {"local_model": UNKNOWN},
            {"cli_provider": "openai", "model_name": UNKNOWN},
            {"cli_base_url": LOCAL, "model_name": UNKNOWN},
        ):
            with self.subTest(selection=selection):
                confirm = mock.Mock()
                app.resolve_backend_target(
                    **selection, config_base_url=LOCAL, confirm_local_fallback=confirm)
                confirm.assert_not_called()

    def test_prompt_receives_actual_host_and_port(self):
        confirm = mock.Mock(return_value=True)
        app.resolve_backend_target(model_name=UNKNOWN, local_host="192.168.1.2",
                                   local_port=9000, confirm_local_fallback=confirm)
        confirm.assert_called_once_with(UNKNOWN, "http://192.168.1.2:9000/v1")

    def test_explicit_destinations_and_known_models_do_not_prompt(self):
        selections = (
            {"model_name": "known"},
            {"model_name": UNKNOWN, "cli_local": True},
            {"local_model": UNKNOWN},
            {"model_name": UNKNOWN, "cli_provider": "openai"},
            {"config_model": UNKNOWN, "config_provider": "openai"},
            {"model_name": UNKNOWN, "cli_base_url": LOCAL},
            {"model_name": UNKNOWN, "config_base_url": "https://proxy.example/v1"},
            {},
        )
        for selection in selections:
            with self.subTest(selection=selection):
                confirm = mock.Mock(side_effect=AssertionError("unexpected prompt"))
                app.resolve_backend_target(**selection, confirm_local_fallback=confirm)
                confirm.assert_not_called()

    def test_online_rejection_does_not_offer_local_fallback(self):
        confirm = mock.Mock()
        with self.assertRaisesRegex(ValueError, "local fallback is disabled"):
            app.resolve_backend_target(model_name=UNKNOWN, cli_online=True,
                                       confirm_local_fallback=confirm)
        confirm.assert_not_called()

    def test_agent_refusal_stops_before_backend_creation(self):
        config = {"system_prompt": "system", "overbudget": "stop"}
        with mock.patch.object(app, "read_configuration", return_value=config), \
             mock.patch.object(app, "load_agent_config", return_value={}), \
             mock.patch.object(app, "confirm_unresolved_model", return_value=False) as confirm, \
             mock.patch.object(app, "create_backend") as backend:
            with self.assertRaisesRegex(ValueError, "fallback cancelled"):
                app.Agent("fake.yaml", "task", model=UNKNOWN, session_id="test")
        confirm.assert_called_once_with(UNKNOWN, LOCAL)
        backend.assert_not_called()

    def test_commit_refusal_stops_before_backend_creation(self):
        with mock.patch.object(commit_cli, "read_configuration", return_value={}), \
             mock.patch.object(commit_cli, "load_agent_config", return_value={}), \
             mock.patch.object(commit_cli, "confirm_unresolved_model", return_value=False) as confirm, \
             mock.patch.object(commit_cli, "create_backend") as backend, \
             mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "fallback cancelled"):
                commit_cli._generate_commit_message("diff", ["file.py"], model=UNKNOWN)
        confirm.assert_called_once_with(UNKNOWN, LOCAL)
        backend.assert_not_called()

    def test_commit_consent_passes_target_to_backend(self):
        with mock.patch.object(commit_cli, "read_configuration", return_value={}), \
             mock.patch.object(commit_cli, "load_agent_config", return_value={}), \
             mock.patch.object(commit_cli, "confirm_unresolved_model", return_value=True), \
             mock.patch.object(commit_cli, "create_backend") as backend, \
             mock.patch.dict("os.environ", {}, clear=True):
            backend.return_value.generate_response.return_value = "Commit message"
            _, model = commit_cli._generate_commit_message("diff", ["file.py"], model=UNKNOWN)
        self.assertEqual(model, UNKNOWN)
        backend.assert_called_once_with("openai", model=UNKNOWN, base_url=LOCAL)


class TestConfirmationPrompt(unittest.TestCase):
    def test_only_explicit_yes_accepts(self):
        for answer in ("y\n", " YES \n", "n\n", "\n", "", "sure\n"):
            with self.subTest(answer=answer):
                out, terminal = io.StringIO(), io.StringIO(answer)
                with mock.patch.object(ui, "_get_tty", return_value=out), \
                     mock.patch("builtins.open", return_value=terminal):
                    accepted = ui.confirm_unresolved_model(UNKNOWN, LOCAL)
                self.assertEqual(accepted, answer.strip().lower() in ("y", "yes"))
                self.assertTrue(terminal.closed)
                self.assertIn(UNKNOWN, out.getvalue())
                self.assertIn(LOCAL, out.getvalue())
                self.assertIn("cannot verify", out.getvalue())
                self.assertIn("[y/N]", out.getvalue())

    def test_piped_input_is_not_consent(self):
        stdin = mock.Mock()
        stdin.isatty.return_value = False
        with mock.patch.object(ui, "_get_tty", return_value=io.StringIO()), \
             mock.patch("builtins.open", side_effect=OSError), \
             mock.patch.object(ui.sys, "stdin", stdin):
            self.assertFalse(ui.confirm_unresolved_model(UNKNOWN, LOCAL))
        stdin.readline.assert_not_called()
        stdin.read.assert_not_called()

    def test_interactive_stdin_fallback(self):
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        stdin.readline.return_value = "yes\n"
        with mock.patch.object(ui, "_get_tty", return_value=io.StringIO()), \
             mock.patch("builtins.open", side_effect=OSError), \
             mock.patch.object(ui.sys, "stdin", stdin):
            self.assertTrue(ui.confirm_unresolved_model(UNKNOWN, LOCAL))
        stdin.close.assert_not_called()

    def test_interrupt_and_read_failure_cancel(self):
        for error in (KeyboardInterrupt, EOFError, OSError):
            with self.subTest(error=error):
                terminal = mock.Mock()
                terminal.readline.side_effect = error
                with mock.patch.object(ui, "_get_tty", return_value=io.StringIO()), \
                     mock.patch("builtins.open", return_value=terminal):
                    self.assertFalse(ui.confirm_unresolved_model(UNKNOWN, LOCAL))
                terminal.close.assert_called_once()
