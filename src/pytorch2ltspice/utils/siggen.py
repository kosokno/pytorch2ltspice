from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence


_DEFAULT_ASY = """Version 4
SymbolType BLOCK
LINE Normal -16 0 -32 -14
LINE Normal -16 0 -32 14
RECTANGLE Normal -32 -24 32 24
TEXT -15 0 Left 2 SG
WINDOW 0 0 -24 Bottom 2
PIN -32 0 NONE 8
PINATTR PinName clk
PINATTR SpiceOrder 1
PIN 32 0 NONE 8
PINATTR PinName out
PINATTR SpiceOrder 2
"""


def _build_siggen_subckt_text(signals: Sequence[float]) -> str:
    """
    Build the LTspice directive text that contains the whole sig_gen subckt.
    Returned string is MULTI-LINE (with '\n'), caller should escape it as '\\n'
    for embedding into .asc TEXT line.
    """
    n = len(signals)
    if n <= 0:
        raise ValueError("signals must be non-empty.")

    lines: list[str] = []
    lines.append("* Auto Generated SigGen subcircuit")
    lines.append(".SUBCKT sig_gen clk out")
    lines.append("XX1 N001 const")
    lines.append("XX2 N001 cnt N002 sum")
    lines.append("XX3 clk N002 cnt sg_samplehold")

    # table(V(cnt), 0,0.0, 1,v1, 2,v2, ...)
    table_args = ["0,0.0"]
    for i, v in enumerate(signals, start=1):
        table_args.append(f"{i},{v:.6f}")
    lines.append("B1 out 0 V=table(V(cnt), " + ", ".join(table_args) + ")")

    # blocks
    lines += [
        "*--- const subcircuit: outputs DC K on its single pin ---",
        ".SUBCKT const K",
        "V1 K 0 {K}",
        ".PARAM K=1",
        ".ENDS const",
        "",
        "*--- sum subcircuit: adds two inputs ---",
        ".SUBCKT sum in1 in2 out",
        "B1 OUT 0 V=V(IN1)+V(IN2)",
        ".ENDS sum",
        "",
        "*--- samplehold subcircuit: latch on rising clk ---",
        ".SUBCKT sg_samplehold CLK IN OUT",
        "R1 o 0 1k",
        "B1 OUT 0 V=V(o)",
        ".machine",
        ".state LO 0",
        ".state LATCH 1",
        ".state HI 2",
        ".rule LO LATCH V(CLK)>=.5",
        ".rule LATCH HI  V(CLK)>=.9",
        ".rule * LO V(CLK)<.5",
        ".output (o) IF((state==1),V(in),V(out))",
        ".endmachine",
        ".ENDS sg_samplehold",
        "",
    ]

    # (optional) export params d1..dN (your original function did this)
    lines += [f".param d{i}={signals[i-1]:.6f}" for i in range(1, n + 1)]

    lines.append(".ENDS SigGen")
    lines.append("")
    lines.append(".backanno")
    lines.append(".end")

    return "\n".join(lines)


def generate_siggen_asc_asy(
    signals: Sequence[float],
    asc_path: str | Path,
    asy_path: Optional[str | Path] = None,
    *,
    gen_symbol: bool = True,
    subckt_name: str = "sig_gen",
) -> tuple[Path, Optional[Path]]:
    """
    Generate:
      - .asc: a minimal schematic that instantiates X1 <subckt_name> clk out
              and embeds the whole .SUBCKT body as a TEXT directive.
      - .asy: (optional) a 2-pin block symbol (clk, out)

    Notes:
      - The generated .asc matches the style of your uploaded sig_gen.asc:
          Version/SHEET/WIRE/FLAG/IOPIN + TEXT for X1 + TEXT for subckt body
      - asy_path defaults to <asc_path>.with_suffix(".asy") if gen_symbol=True
    """
    asc_path = Path(asc_path)
    if gen_symbol and asy_path is None:
        asy_path = asc_path.with_suffix(".asy")
    asy_path = Path(asy_path) if asy_path is not None else None

    # Build embedded subckt text (multi-line) then escape for .asc TEXT line
    subckt_text = _build_siggen_subckt_text(signals)
    subckt_text_escaped = subckt_text.replace("\n", "\\n")

    # Minimal .asc (coordinates taken from your sample)
    # NOTE: "!" indicates a Spice directive in LTspice schematic TEXT lines.
    asc_lines = [
        "Version 4.1",
        "SHEET 1 381476 58612",
        "WIRE 176 224 112 224",
        "WIRE 368 224 320 224",
        "FLAG 112 224 clk",
        "IOPIN 112 224 In",
        "FLAG 368 224 out",
        "IOPIN 368 224 Out",
        f"TEXT 48 144 Left 2 !X1 {subckt_name} clk out",
        f"TEXT 40 272 Left 2 !{subckt_text_escaped}",
        "",
    ]

    asc_path.parent.mkdir(parents=True, exist_ok=True)
    asc_path.write_text("\n".join(asc_lines), encoding="utf-8")

    written_asy: Optional[Path] = None
    if gen_symbol:
        assert asy_path is not None
        asy_path.parent.mkdir(parents=True, exist_ok=True)
        asy_path.write_text(_DEFAULT_ASY, encoding="utf-8")
        written_asy = asy_path

    return asc_path, written_asy
