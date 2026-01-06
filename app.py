import streamlit as st
from Bio.Seq import Seq
import re
import urllib.parse

st.set_page_config(page_title="HA/NA Sequence Quick-Check", layout="centered")

st.title("HA/NA Sequence Identity Quick-Check (QC)")
st.caption("Screening tool for sequence sanity check + quick link to NCBI BLAST. Final QC decisions must rely on validated assays.")

# ------------------ UI inputs ------------------
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)

sequence = st.text_area("Paste nucleotide sequence (FASTA or raw):", height=260)

# ------------------ helper functions ------------------
def clean_sequence(seq: str) -> str:
    """Remove FASTA headers and non-ATGC chars; keep only ATGC."""
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return (gc / len(seq)) * 100

def translate_frame(nt: str, frame: int) -> str:
    """Translate a nucleotide sequence in a given frame (0,1,2)."""
    trimmed = nt[frame:]
    trimmed = trimmed[: (len(trimmed) // 3) * 3]
    if len(trimmed) == 0:
        return ""
    return str(Seq(trimmed).translate(to_stop=False))

def best_orf_6frames(nt: str):
    """
    Pick best reading frame among 6 frames:
    - maximize AA length before first stop
    - minimize number of stops
    """
    candidates = []
    fwd = nt
    rev = str(Seq(nt).reverse_complement())

    for strand_name, seq in [("forward", fwd), ("reverse_complement", rev)]:
        for frame in [0, 1, 2]:
            aa = translate_frame(seq, frame)
            pre_stop = aa.split("*")[0] if aa else ""
            stop_count = aa.count("*") if aa else 0
            candidates.append({
                "strand": strand_name,
                "frame": frame + 1,
                "aa": aa,
                "aa_len_total": len(aa),
                "aa_len_before_first_stop": len(pre_stop),
                "stop_count": stop_count,
            })

    best = sorted(
        candidates,
        key=lambda x: (x["aa_len_before_first_stop"], -x["stop_count"], x["aa_len_total"]),
        reverse=True
    )[0]
    return best, candidates

def build_blast_url(nt_sequence: str) -> str:
    """
    Build NCBI BLAST URL with prefilled QUERY.
    Note: very long sequences may exceed URL length limits; works best for short/medium fragments.
    """
    base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    params = {
        "PROGRAM": "blastn",
        "PAGE_TYPE": "BlastSearch",
        "QUERY": nt_sequence
    }
    return base_url + "?" + urllib.parse.urlencode(params)

# ------------------ main action ------------------
if st.button("Analyze Sequence", type="primary"):
    if not sequence.strip():
        st.error("Please paste a sequence.")
        st.stop()

    nt = clean_sequence(sequence)

    if len(nt) == 0:
        st.error("No valid A/T/G/C bases found after cleaning. Please check input.")
        st.stop()

    length = len(nt)
    gc = gc_content(nt)

    best, all_frames = best_orf_6frames(nt)

    # QC-oriented heuristics (tune thresholds based on your typical amplicon length)
    early_stop_flag = best["aa_len_before_first_stop"] < 80
    many_stops_flag = best["stop_count"] > 1

    st.subheader("🔍 Analysis Result")
    st.write(f"Sample ID: **{sample_id if sample_id else '-'}**")
    st.write(f"Expected gene: **{expected_gene}**")
    st.write(f"Sequence length: **{length} bp**")
    st.write(f"GC content: **{gc:.1f}%**")

    st.markdown("### ORF / Translation (6-frame scan)")
    st.write(
        f"Best frame: **{best['strand']}**, frame **{best['frame']}** "
        f"(AA length before first stop: **{best['aa_len_before_first_stop']}**, total stops: **{best['stop_count']}**)"
    )

    if early_stop_flag or many_stops_flag:
        st.warning("ORF check: ⚠️ INVESTIGATE (early stop codon or multiple stops).")
    else:
        st.success("ORF check: ✅ PASS (no early stop issues in best frame).")

    with st.expander("Show translated AA (best frame)"):
        st.code(best["aa"] if best["aa"] else "(empty)", language="text")

    with st.expander("Show 6-frame summary"):
        for c in all_frames:
            st.write(
                f"- {c['strand']} frame {c['frame']}: "
                f"before first stop = {c['aa_len_before_first_stop']} aa, stops = {c['stop_count']}"
            )

    # BLAST link
    st.markdown("### 🔗 External Analysis")
    blast_url = build_blast_url(nt)
    st.markdown(f"[👉 Run NCBI BLAST (blastn)]({blast_url})")

    # QC Assessment (Phase 1)
    st.markdown("### QC Assessment (Phase 1)")
    st.info(
        "This app performs sequence sanity checks (length/GC/ORF). "
        "For HA/NA identity and subtype determination, click the BLAST link and review top hits. "
        "Use validated wet-lab methods for final QC decisions."
    )

st.divider()
st.caption("Tip: If your sequence is very long, the BLAST URL may be too long. In that case, open BLAST and paste the sequence manually.")
