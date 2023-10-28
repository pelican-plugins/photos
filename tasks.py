from inspect import cleandoc
import os
from pathlib import Path
from shutil import which
import sys

from invoke import task

PKG_NAME = "photos"
PKG_PATH = Path(f"pelican/plugins/{PKG_NAME}")
TOOLS = ("poetry", "pre-commit")

ACTIVE_VENV = os.environ.get("VIRTUAL_ENV", None)
VENV_HOME = Path(os.environ.get("WORKON_HOME", "~/.local/share/virtualenvs"))
VENV_PATH = Path(ACTIVE_VENV) if ACTIVE_VENV else (VENV_HOME.expanduser() / PKG_NAME)
VENV = str(VENV_PATH.expanduser())
BIN_DIR = "bin" if os.name != "nt" else "Scripts"
VENV_BIN = Path(VENV) / Path(BIN_DIR)
POETRY = which("poetry") if which("poetry") else (VENV_BIN / "poetry")
CMD_PREFIX = f"{VENV_BIN}/" if ACTIVE_VENV else f"{POETRY} run "
PRECOMMIT = which("pre-commit") if which("pre-commit") else f"{CMD_PREFIX}pre-commit"
PTY = os.name != "nt"


@task
def tests(c):
    """Run the test suite."""
    c.run(f"{CMD_PREFIX}pytest", pty=PTY)


@task
def black(c, check=False, diff=False):
    """Run Black auto-formatter, optionally with `--check` or `--diff`."""
    check_flag, diff_flag = "", ""
    if check:
        check_flag = "--check"
    if diff:
        diff_flag = "--diff"
    c.run(f"{CMD_PREFIX}black {check_flag} {diff_flag} {PKG_PATH} tasks.py", pty=PTY)


@task
def ruff(c, fix=False, diff=False):
    """Run Ruff to ensure code meets project standards."""
    diff_flag, fix_flag = "", ""
    if fix:
        fix_flag = "--fix"
    if diff:
        diff_flag = "--diff"
    c.run(f"{CMD_PREFIX}ruff check {diff_flag} {fix_flag} .", pty=PTY)


@task
def lint(c, fix=False, diff=False):
    """Check code style via linting tools."""
    ruff(c, fix=fix, diff=diff)
    black(c, check=(not fix), diff=diff)


@task
def tools(c):
    """Install development tools in the virtual environment if not already on PATH."""
    for tool in TOOLS:
        if not which(tool):
            c.run(f"{CMD_PREFIX}pip install {tool}")


@task
def precommit(c):
    """Install pre-commit hooks to .git/hooks/pre-commit."""
    c.run(f"{PRECOMMIT} install")


@task
def setup(c):
    """Set up the development environment."""
    if which("poetry") or ACTIVE_VENV:
        tools(c)
        c.run(f"{CMD_PREFIX}python -m pip install --upgrade pip")
        c.run(f"{POETRY} install")
        precommit(c)
        print(  # noqa: T201
            "\nDevelopment environment should now be set up and ready!\n"
        )
    else:
        error_message = """
            Poetry is not installed, and there is no active virtual environment available.
            You can either manually create and activate a virtual environment, or you can
            install Poetry via:

            curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

            Once you have taken one of the above two steps, run `invoke setup` again.
            """  # noqa: E501
        sys.exit(cleandoc(error_message))
