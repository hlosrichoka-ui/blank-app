# app.py
import re
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="NT → Protein + Protein Visualization", layout="centered")
st.title("NT → Protein Translator + Protein Visualization (QC-friendly)")

# =========================================================
# Helpers
# =========================================================
AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

def clean_nt_keep_len(seq: str) -> str:
    """
    - Remove FASTA headers
    - Keep letters only
    - Convert non-ATGC to N (do NOT delete)
    """
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    s = "".join(lines).upper()
    s = re.sub(r"[^A-Z]", "", s)
    s = re.sub(r"[^ATGC]", "N", s)
    return s

def translate_frame(nt: str, strand: str, frame: int) -> str:
    """
    strand: '+' or '-'
    frame: 0,1,2 (0-based)
    Translate including '*' stops.
    """
    if strand == "-":
        nt = str(Seq(nt).reverse_complement())
    trimmed = nt[frame:]
    trimmed = trimmed[: (len(trimmed) // 3) * 3]
    if not trimmed:
        return ""
    aa = str(Seq(trimmed).translate(to_stop=False))
    return aa

def find_best_orf(nt: str):
    """
    Pick best ORF across 6 frames by:
      1) longest AA segment before first stop
      2) if tie, fewer total stops in full translation
    Returns dict with best strand/frame, aa_full, aa_orf (pre-stop), start_nt_index (0-based in oriented strand)
    """
    best = {
        "strand": "+",
        "frame": 0,
        "orf_len": 0,
        "stop_count": 10**9,
        "aa_full": "",
        "aa_orf": "",
    }

    for strand in ["+", "-"]:
        for frame in [0, 1, 2]:
            aa_full = translate_frame(nt, strand, frame)
            if not aa_full:
                continue
            aa_orf = aa_full.split("*")[0]
            orf_len = len(aa_orf)
            stop_count = aa_full.count("*")

            if (orf_len > best["orf_len"]) or (orf_len == best["orf_len"] and stop_count < best["stop_count"]):
                best.update(
                    strand=strand,
                    frame=frame,
                    orf_len=orf_len,
                    stop_count=stop_count,
                    aa_full=aa_full,
                    aa_orf=aa_orf,
                )
    return best

def sanitize_protein(aa: str) -> str:
    """
    Keep only standard 20 AAs; drop '*' and unknowns.
    """
    aa = aa.upper()
    return "".join([c for c in aa if c in AA_VALID])

def kyte_doolittle_hydropathy(aa: str, window: int = 19):
    """
    Simple Kyte-Doolittle hydropathy (sliding window average).
    Returns positions (1-based) and values.
    """
    kd = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
        "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
        "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
    }
    aa = sanitize_protein(aa)
    if len(aa) < window:
        return [], []

    vals = []
    pos = []
    half = window // 2
    for i in range(half, len(aa) - half):
        seg = aa[i - half : i + half + 1]
        v = sum(kd.get(x, 0.0) for x in seg) / window
        pos.append(i + 1)  # 1-based
        vals.append(v)
    return pos, vals

def predict_tm_segments(pos, vals, threshold=1.6, min_len=18):
    """
    Heuristic TM prediction:
      contiguous region where hydropathy >= threshold for at least min_len points
    Returns list of (start_aa, end_aa) 1-based approx.
    """
    segs = []
    if not pos:
        return segs

    in_seg = False
    start = None
    last_pos = None
    for p, v in zip(pos, vals):
        if v >= threshold:
            if not in_seg:
                in_seg = True
                start = p
            last_pos = p
        else:
            if in_seg:
                end = last_pos
                if end is not None and start is not None and (end - start + 1) >= min_len:
                    segs.append((start, end))
                in_seg = False
                start = None
                last_pos = None

    if in_seg and start is not None and last_pos is not None:
        if (last_pos - start + 1) >= min_len:
            segs.append((start, last_pos))
    return segs

def draw_protein_schematic(length: int, tm_segments):
    """
    Simple 2D schematic: a line representing protein length + highlight TM segments as blocks.
    """
    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.set_xlim(0, length)
    ax.set_ylim(0, 1)
    ax.hlines(0.5, 0, length, linewidth=6)
    for (s, e) in tm_segments:
        ax.add_patch(plt.Rectangle((s, 0.35), e - s + 1, 0.3))
    ax.set_yticks([])
    ax.set_xlabel("Amino acid position (1-based)")
    ax.set_title("Protein schematic (TM-like hydrophobic segments highlighted)")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    return fig

# =========================================================
# UI
# =========================================================
sample_id = st.text_input("Sample ID")

nt_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=240)

mode = st.radio(
    "Translation mode",
    ["Auto (best ORF across 6 frames)", "Manual (choose strand + frame)"],
    horizontal=False
)

colA, colB = st.columns(2)
with colA:
    strand = st.selectbox("Strand", ["+", "-"], disabled=(mode.startswith("Auto")))
with colB:
    frame = st.selectbox("Frame", [0, 1, 2], disabled=(mode.startswith("Auto")))

st.markdown("---")
st.subheader("Protein visualization settings")
window = st.slider("Hydropathy window size", min_value=7, max_value=31, value=19, step=2)
tm_threshold = st.slider("Hydrophobic threshold (TM heuristic)", min_value=0.5, max_value=3.0, value=1.6, step=0.1)
tm_min_len = st.slider("Minimum length for TM segment", min_value=10, max_value=30, value=18, step=1)

# =========================================================
# Analyze
# =========================================================
if st.button("Translate + Visualize", type="primary"):
    NT = clean_nt_keep_len(nt_raw)
    if not NT:
        st.error("Please paste a valid nucleotide sequence.")
        st.stop()

    if mode.startswith("Auto"):
        best = find_best_orf(NT)
        strand_use = best["strand"]
        frame_use = best["frame"]
        aa_full = best["aa_full"]
        aa_orf = best["aa_orf"]
        stop_count = best["stop_count"]
        orf_len = best["orf_len"]
        translation_note = f"Auto-selected best ORF: strand {strand_use}, frame {frame_use} (ORF={orf_len} aa, stops in full translation={stop_count})"
    else:
        strand_use = strand
        frame_use = int(frame)
        aa_full = translate_frame(NT, strand_use, frame_use)
        aa_orf = aa_full.split("*")[0] if aa_full else ""
        stop_count = aa_full.count("*") if aa_full else 0
        orf_len = len(aa_orf)
        translation_note = f"Manual translation: strand {strand_use}, frame {frame_use} (ORF={orf_len} aa, stops in full translation={stop_count})"

    aa_clean = sanitize_protein(aa_orf)

    if not aa_clean:
        st.error("Could not generate a protein sequence (check input length / frame / ambiguity).")
        st.stop()

    # Protein analysis
    pa = ProteinAnalysis(aa_clean)
    aa_comp = pa.get_amino_acids_percent()  # fraction
    mw = pa.molecular_weight()
    arom = pa.aromaticity()
    gravy = pa.gravy()
    iso = pa.isoelectric_point()

    # Hydropathy + TM heuristic
    pos, vals = kyte_doolittle_hydropathy(aa_clean, window=window)
    tm_segments = predict_tm_segments(pos, vals, threshold=tm_threshold, min_len=tm_min_len)

    # =====================================================
    # Display results
    # =====================================================
    st.markdown("## 🔍 Result")
    st.write(f"**Sample ID:** {sample_id if sample_id else '-'}")
    st.write(f"**Input length:** {len(NT)} nt")
    st.write(f"**Translation:** {translation_note}")
    st.write(f"**Protein length (ORF, before first stop):** {len(aa_clean)} aa")

    st.markdown("### Protein quick properties")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MW (Da)", f"{mw:,.1f}")
    c2.metric("pI", f"{iso:.2f}")
    c3.metric("Aromaticity", f"{arom:.3f}")
    c4.metric("GRAVY", f"{gravy:.3f}")

    st.markdown("### Protein sequence (FASTA)")
    prot_fasta = f">{sample_id or 'protein'}|strand={strand_use}|frame={frame_use}\n{aa_clean}\n"
    st.code(prot_fasta, language="text")

    # =====================================================
    # Visualizations
    # =====================================================
    st.markdown("### 📈 Visualizations")

    # 1) Hydropathy plot
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    if pos:
        ax1.plot(pos, vals)
        ax1.axhline(tm_threshold, linestyle="--")
        ax1.set_xlabel("Amino acid position (1-based)")
        ax1.set_ylabel("Hydropathy (Kyte-Doolittle)")
        ax1.set_title(f"Hydropathy plot (window={window})")
    else:
        ax1.text(0.5, 0.5, "Protein too short for hydropathy window", ha="center", va="center")
        ax1.set_axis_off()
    st.pyplot(fig1, clear_figure=True)

    # 2) Amino acid composition bar chart
    comp_df = pd.DataFrame({
        "AA": list(aa_comp.keys()),
        "Percent": [aa_comp[k] * 100 for k in aa_comp.keys()]
    }).sort_values("AA")

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.bar(comp_df["AA"], comp_df["Percent"])
    ax2.set_xlabel("Amino acid")
    ax2.set_ylabel("Percent (%)")
    ax2.set_title("Amino acid composition (%)")
    st.pyplot(fig2, clear_figure=True)

    # 3) Protein schematic with TM-like segments
    st.markdown("### 🧬 Protein schematic (heuristic)")
    if tm_segments:
        st.write("**Predicted TM-like segments (heuristic):** " + ", ".join([f"{s}-{e}" for s, e in tm_segments]))
    else:
        st.write("**Predicted TM-like segments (heuristic):** -")
    fig3 = draw_protein_schematic(len(aa_clean), tm_segments)
    st.pyplot(fig3, clear_figure=True)

    # =====================================================
    # Downloads
    # =====================================================
    st.markdown("### ⬇️ Downloads")

    # Protein FASTA
    st.download_button(
        label="Download protein FASTA",
        data=prot_fasta.encode("utf-8"),
        file_name=f"{sample_id or 'protein'}_protein.fasta",
        mime="text/plain",
    )

    # Summary CSV
    summary = pd.DataFrame([{
        "Sample ID": sample_id,
        "Input length (nt)": len(NT),
        "Strand": strand_use,
        "Frame": frame_use,
        "Protein length (aa)": len(aa_clean),
        "Stops in full translation": stop_count,
        "MW (Da)": round(mw, 1),
        "pI": round(iso, 2),
        "Aromaticity": round(arom, 3),
        "GRAVY": round(gravy, 3),
        "Hydropathy window": window,
        "TM threshold": tm_threshold,
        "TM min length": tm_min_len,
        "Predicted TM segments": ", ".join([f"{s}-{e}" for s, e in tm_segments]) if tm_segments else "",
    }])

    st.download_button(
        label="Download summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_id or 'protein'}_protein_summary.csv",
        mime="text/csv",
    )

st.caption(
    "Note: 'Protein visualization' here is a QC-friendly 2D simulation (hydropathy, composition, schematic). "
    "It is not a true 3D structure prediction like AlphaFold."
)
