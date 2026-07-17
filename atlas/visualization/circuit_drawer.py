"""Circuit drawing, kept separate from ansatz construction and VQE execution.

This module is a thin wrapper around Qiskit's ``circuit.draw()``. Atlas keeps
drawing here (rather than inside ansatz builders or experiment runners) so that
visualization stays optional and never couples into quantum-circuit construction
or optimization loops.

Implementation choice: Qiskit's ``mpl`` drawer returns a
``matplotlib.figure.Figure`` object rather than writing a file directly, so for
``output_format="mpl"`` (the default) we save that figure via
``Figure.savefig()``. For text-based or other formats where Qiskit's
``circuit.draw()`` accepts a ``filename`` argument and writes the file itself,
we delegate directly to ``circuit.draw(..., filename=...)`` instead of
re-implementing file output. This keeps the function a thin, robust wrapper
around whichever save path Qiskit already supports for the requested format.
"""

from __future__ import annotations


def draw_circuit(circuit, output_path: str, output_format: str = "mpl") -> None:
    """Render a quantum circuit image (or text drawing) to disk.

    Purpose:
        Persist a visual representation of ``circuit`` at ``output_path`` using
        Qiskit's drawer, so notebooks and experiment pipelines can dump circuit
        diagrams without owning matplotlib/Qiskit save details.

    Inputs:
        circuit: A Qiskit ``QuantumCircuit`` (or any object exposing
            ``.draw(...)`` with the same signature).
        output_path: Filesystem path where the drawing should be written
            (e.g. ``"artifacts/ansatz.png"``). Parent directories are assumed to
            already exist; this function does not create them.
        output_format: Qiskit drawer backend name. Default ``"mpl"`` produces a
            matplotlib figure. Other values (e.g. ``"text"``, ``"latex"``) are
            forwarded to Qiskit and use its ``filename`` write path.

    Process:
        1. If ``output_format == "mpl"``, call ``circuit.draw(output="mpl")`` to
           obtain a ``Figure``, then call ``savefig`` with a high DPI and tight
           bounding box so the exported PNG is publication-ready.
        2. Otherwise, call ``circuit.draw(output=..., filename=output_path)`` and
           let Qiskit write the file itself (formats that already know how to
           serialize to disk).

    Outputs:
        None. The drawing is written to ``output_path`` as a side effect.

    Side Effects:
        Creates or overwrites the file at ``output_path``. For ``mpl``, may
        allocate a matplotlib figure that is not explicitly closed here (Qiskit's
        drawer owns that lifecycle).
    """

    if output_format == "mpl":
        # mpl returns an in-memory Figure; Qiskit does not write the file for us.
        figure = circuit.draw(output=output_format)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        # Non-mpl formats already accept filename= and write through Qiskit.
        circuit.draw(output=output_format, filename=output_path)
