import re
import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="NT → Protein Analysis & Visualization",
    layout="centered"
)
st.title("NT → Protein Analysis & Visualization")

# =========================================================
# Helper functions
# =========================================================
AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

def clean_nt(seq: str) -> str:
    """Remove FASTA header, keep length, convert non-ATGC to N"""
    lines = [l.strip() for l in seq.splitlines() if l and not l.startswith(">")]
    s = "".join(lines).upper()
    s = re.sub(r"[^A-Z]", "", s)
    s = re.sub(r"[^ATGC]", "N", s)
    return s

def translate_frame(nt: str, strand: str, frame: int) -> str:
    if strand == "-":
        nt = str(Seq(nt).reverse_complement())
    nt = nt[frame:]
    nt = nt[: (len(nt)//3)*3]
    return str(Seq(nt).translate(to_stop=False)) if nt else ""

def best_orf_6frames(nt: str):
    best = {"strand": "+", "frame": 0, "aa": "", "len": 0}
    for strand in ["+", "-"]:
        for frame in [0,1,2]:
            aa = translate_frame(nt, strand, frame)
            orf = aa.split("*")[0]
            if len(orf) > best["len"]:
                best = {
                    "strand": strand,
                    "frame": frame,
                    "aa": orf,
                    "len": len(orf)
                }
    return best

def sanitize_protein(aa: str) -> str:
    return "".join([c for c in aa if c in AA_VALID])

def kyte_doolittle(aa: str, window=19):
    kd = {
        "I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,
        "G":-0.4,"T":-0.7,"S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,
        "H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5
    }
    if len(aa) < window:
        return []
    vals = []
    for i in range(len(aa)-window+1):
        seg = aa[i:i+window]
        vals.append(sum(kd[a] for a in seg)/window)
    return vals

def predict_tm(vals, threshold=1.6, min_len=18):
    segs, start = [], None
    for i,v in enumerate(vals):
        if v >= threshold:
            if start is None:
                start = i+1
        else:
            if start and (i+1-start)>=min_len:
                segs.append((start, i))
            start = None
    if start and (len(vals)+1-start)>=min_len:
        segs.append((start, len(vals)))
    return segs

# =========================================================
# UI
# =========================================================
sample_id = st.text_input("Sample ID")
nt_raw = st.text_area("Paste nucleotide sequence (FASTA)", height=220)

mode = st.radio(
    "Translation mode",
    ["Auto (best ORF)", "Manual"],
    horizontal=True
)

col1,col2 = st.columns(2)
strand = col1.selectbox("Strand", ["+","-"], disabled=(mode=="Auto (best ORF)"))
frame = col2.selectbox("Frame", [0,1,2], disabled=(mode=="Auto (best ORF)"))

window = st.slider("Hydropathy window", 7, 31, 19, step=2)
tm_threshold = st.slider("TM threshold", 0.5, 3.0, 1.6, step=0.1)

# =========================================================
# Run
# =========================================================
if st.button("Translate & Analyze", type="primary"):
    NT = clean_nt(nt_raw)
    if not NT:
        st.error("Invalid nucleotide sequence")
        st.stop()

    if mode == "Auto (best ORF)":
        best = best_orf_6frames(NT)
        aa = best["aa"]
        strand_use = best["strand"]
        frame_use = best["frame"]
    else:
        aa = translate_frame(NT, strand, frame).split("*")[0]
        strand_use = strand
        frame_use = frame

    aa = sanitize_protein(aa)

    if not aa:
        st.error("Protein translation failed")
        st.stop()

    pa = ProteinAnalysis(aa)
    hydropathy = kyte_doolittle(aa, window)
    tm_segs = predict_tm(hydropathy, tm_threshold)

    # =====================================================
    # Results
    # =====================================================
    st.markdown("## 🔬 Results")
    st.write(f"**Sample:** {sample_id or '-'}")
    st.write(f"**Protein length:** {len(aa)} aa")
    st.write(f"**Strand / Frame:** {strand_use} / {frame_use}")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MW (Da)", f"{pa.molecular_weight():,.1f}")
    c2.metric("pI", f"{pa.isoelectric_point():.2f}")
    c3.metric("GRAVY", f"{pa.gravy():.2f}")
    c4.metric("Aromaticity", f"{pa.aromaticity():.3f}")

    # =====================================================
    # Visualization (NO matplotlib)
    # =====================================================
    st.markdown("## 📈 Hydropathy plot")
    if hydropathy:
        df = pd.DataFrame({"Hydropathy": hydropathy})
        st.line_chart(df)
    else:
        st.info("Protein too short for hydropathy window")

    st.markdown("## 🧬 TM-like segments (heuristic)")
    if tm_segs:
        st.write(", ".join([f"{s}-{e}" for s,e in tm_segs]))
    else:
        st.write("None")

    # simple schematic (text-based)
    st.markdown("### Protein schematic")
    bar = ["─"] * len(aa)
    for s,e in tm_segs:
        for i in range(s-1, min(e,len(bar))):
            bar[i] = "█"
    st.code("".join(bar))

    # =====================================================
    # Outputs
    # =====================================================
    st.markdown("## ⬇️ Downloads")

    fasta = f">{sample_id or 'protein'}|strand={strand_use}|frame={frame_use}\n{aa}\n"
    st.download_button(
        "Download protein FASTA",
        fasta.encode(),
        file_name=f"{sample_id or 'protein'}_protein.fasta"
    )

    summary = pd.DataFrame([{
        "Sample ID": sample_id,
        "Protein length (aa)": len(aa),
        "Strand": strand_use,
        "Frame": frame_use,
        "MW": round(pa.molecular_weight(),1),
        "pI": round(pa.isoelectric_point(),2),
        "GRAVY": round(pa.gravy(),2),
        "TM segments": ", ".join([f"{s}-{e}" for s,e in tm_segs])
    }])

    st.download_button(
        "Download summary CSV",
        summary.to_csv(index=False).encode(),
        file_name=f"{sample_id or 'protein'}_summary.csv"
    )

st.caption(
    "Protein visualization here is a QC-friendly simulation (hydropathy + TM heuristic), "
    "not a 3D structure prediction (e.g. AlphaFold)."
)
