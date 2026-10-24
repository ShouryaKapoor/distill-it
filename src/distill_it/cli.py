"""Command-line interface. Subcommands (run, label, train, ...) arrive in later phases."""

import typer

from distill_it import __version__

app = typer.Typer(help="Distil an LLM classifier into a small, fast student model.")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    if version:
        typer.echo(f"distill-it {__version__}")
        raise typer.Exit()
