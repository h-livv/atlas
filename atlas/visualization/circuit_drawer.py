"""Circuit drawing, kept separate from ansatz construction and VQE execution.

Implementation choice: Qiskit's `mpl` drawer returns a `matplotlib.figure.Figure`
object rather than writing a file directly, so for `output_format="mpl"` (the
default) we save that figure via `Figure.savefig()`. For text-based or other
formats where Qiskit's `circuit.draw()` accepts a `filename` argument and
writes the file itself, we delegate directly to `circuit.draw(..., filename=...)`
instead of re-implementing file output. This keeps the function a thin,
robust wrapper around whichever save path Qiskit already supports for the
requested format.
"""

from __future__ import annotations


def draw_circuit(circuit, output_path: str, output_format: str = "mpl") -> None:
    """Render `circuit` to `output_path` using Qiskit's `circuit.draw()`.

    `output_format="mpl"` returns a Figure that is saved with `savefig()`;
    other formats (e.g. `"latex"`, `"text"`) are passed Qiskit's own
    `filename` argument since Qiskit already knows how to write those to disk.
    """

    if output_format == "mpl":
        figure = circuit.draw(output=output_format)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        circuit.draw(output=output_format, filename=output_path)
