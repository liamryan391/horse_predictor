from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


def log_step(label: str, verbose: bool) -> None:
    if verbose:
        print(f"Visual smoke: {label}", flush=True)


def agent_browser_command(capture_output: bool = True) -> list[str]:
    executable = shutil.which("agent-browser")
    if executable is None:
        raise RuntimeError("agent-browser is not installed or not on PATH.")
    if os.name == "nt" and not capture_output:
        cmd_executable = shutil.which("agent-browser.cmd")
        if cmd_executable:
            return ["cmd.exe", "/d", "/c", cmd_executable]
    if os.name == "nt":
        native_executable = (
            Path(executable).parent
            / "node_modules"
            / "agent-browser"
            / "bin"
            / "agent-browser-win32-x64.exe"
        )
        if native_executable.exists():
            return [str(native_executable)]
    return [executable]


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    process.kill()


def run_agent(args: Iterable[str], timeout: float, capture_output: bool = True) -> str:
    command = [*agent_browser_command(capture_output), *args]
    stdout_target = subprocess.PIPE if capture_output else subprocess.DEVNULL
    stderr_target = subprocess.PIPE if capture_output else subprocess.DEVNULL
    process = subprocess.Popen(command, stdout=stdout_target, stderr=stderr_target, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_process_tree(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    if not capture_output:
        return ""
    return (stdout or "") + (stderr or "")


def eval_js(script: str, timeout: float) -> str:
    return run_agent(["eval", script], timeout)


def require_output(output: str, expected: str, message: str) -> None:
    if expected not in output:
        raise RuntimeError(f"{message} Output was: {output.strip()}")


def wait_for_render(timeout: float) -> None:
    run_agent(["wait", "1500"], timeout, capture_output=False)


def click_button(label: str, timeout: float) -> None:
    script = (
        "(() => { const button = Array.from(document.querySelectorAll('button'))"
        f".find((item) => item.textContent && item.textContent.includes({label!r}));"
        "if (!button) { return 'MISSING_BUTTON'; } button.click(); return 'CLICKED'; })()"
    )
    require_output(eval_js(script, timeout), "CLICKED", f"{label} button was not found.")
    run_agent(["wait", "750"], timeout, capture_output=False)


def assert_body_contains(text: str, timeout: float) -> None:
    output = eval_js(f"document.body.innerText.includes({text!r}) ? 'FOUND' : document.body.innerText", timeout)
    require_output(output, "FOUND", f"Expected page text {text!r} was not visible.")


def run_visual_smoke(args: argparse.Namespace) -> None:
    if shutil.which("agent-browser") is None:
        if args.allow_missing:
            print("agent-browser is not installed; visual smoke skipped because --allow-missing was set.")
            return
        raise RuntimeError("agent-browser is not installed or not on PATH.")

    base_url = args.base_url.rstrip("/")
    try:
        log_step("open frontend", args.verbose)
        run_agent(["open", base_url], args.timeout, capture_output=False)
        wait_for_render(args.timeout)
        log_step("check rendered content", args.verbose)
        require_output(
            eval_js("document.body.innerText.trim().length > 0 ? 'HAS_CONTENT' : 'BLANK'", args.timeout),
            "HAS_CONTENT",
            "Frontend page was blank.",
        )
        log_step("check framework overlays", args.verbose)
        require_output(
            eval_js(
                "document.querySelector('.vite-error-overlay, [data-nextjs-dialog], #webpack-dev-server-client-overlay')"
                " ? 'ERROR_OVERLAY' : 'OK'",
                args.timeout,
            ),
            "OK",
            "Frontend displayed a framework error overlay.",
        )

        log_step("navigate workspace", args.verbose)
        click_button("Workspace", args.timeout)
        assert_body_contains("Today's racing desk", args.timeout)
        log_step("navigate race card", args.verbose)
        click_button("Race Card", args.timeout)
        assert_body_contains("SURFACE", args.timeout)
        log_step("navigate evaluation", args.verbose)
        click_button("Evaluation", args.timeout)
        assert_body_contains("Model Evaluation", args.timeout)
        log_step("navigate bet journal", args.verbose)
        click_button("Bet Journal", args.timeout)
        assert_body_contains("Position ledger", args.timeout)
        log_step("navigate responsible use", args.verbose)
        click_button("Responsible Use", args.timeout)
        assert_body_contains("Decision support, not certainty", args.timeout)

        if args.admin_token:
            log_step("set admin credentials", args.verbose)
            eval_js(
                "localStorage.setItem('horse-predictor-access-token', "
                f"{args.admin_token!r}); localStorage.setItem('horse-predictor-access-actor', {args.admin_actor!r}); 'OK'",
                args.timeout,
            )
            log_step("navigate admin", args.verbose)
            click_button("Admin", args.timeout)
            assert_body_contains("Governance desk", args.timeout)
            assert_body_contains("Audit History", args.timeout)
            assert_body_contains("Governed Actions", args.timeout)
        elif args.require_admin:
            raise RuntimeError("--require-admin needs --admin-token or API_AUTH_TOKEN.")

        if not args.skip_mobile:
            log_step("check mobile viewport", args.verbose)
            run_agent(["set", "viewport", str(args.mobile_width), str(args.mobile_height)], args.timeout, capture_output=False)
            run_agent(["reload"], args.timeout, capture_output=False)
            wait_for_render(args.timeout)
            require_output(
                eval_js("document.body.innerText.trim().length > 0 ? 'HAS_CONTENT' : 'BLANK'", args.timeout),
                "HAS_CONTENT",
                "Frontend page was blank at the mobile viewport.",
            )
            click_button("Workspace", args.timeout)
            assert_body_contains("Today's racing desk", args.timeout)

        if args.screenshot:
            log_step("capture screenshot", args.verbose)
            output = run_agent(["screenshot", "--annotate"], args.timeout)
            print(output.strip())
    finally:
        if not args.keep_open:
            try:
                log_step("close browser", args.verbose)
                run_agent(["close"], args.timeout, capture_output=False)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent-browser visual smoke check for Horse Predictor.")
    parser.add_argument("--base-url", default=os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:5173"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--admin-token", default=os.getenv("API_AUTH_TOKEN", ""))
    parser.add_argument("--admin-actor", default=os.getenv("ADMIN_ACTOR", "visual-smoke"))
    parser.add_argument("--require-admin", action="store_true", help="Fail unless the Admin tab is checked with a token.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip instead of failing when agent-browser is unavailable.")
    parser.add_argument("--screenshot", action="store_true", help="Capture an annotated screenshot at the final state.")
    parser.add_argument("--skip-mobile", action="store_true", help="Skip the mobile viewport content check.")
    parser.add_argument("--mobile-width", type=int, default=390)
    parser.add_argument("--mobile-height", type=int, default=844)
    parser.add_argument("--keep-open", action="store_true", help="Leave the browser session open after the smoke check.")
    parser.add_argument("--verbose", action="store_true", help="Print progress for each browser smoke step.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_visual_smoke(args)
    print(f"Visual smoke check passed for {args.base_url.rstrip('/')}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
