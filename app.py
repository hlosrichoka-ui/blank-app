import streamlit as st
from Bio.Seq import Seq
import re

st.set_page_config(page_title="HA/NA Sequence Quick-Check")

st.title("HA/NA Sequence Identity Quick-Check (QC)")

sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"])

sequence = st.text_area("Paste nucleotide sequence (FASTA or raw):", height=250)

def clean_sequence(seq: str) -> str:
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return (gc / len(seq)) * 100

def translate_frame(nt: str, frame: int) -> str:
    trimmed = nt[frame:]
    trimmed = trimmed[: (len(trimmed) // 3) * 3]
    return str(Seq(trimmed).translate(to_stop=False))

def best_orf_6frames(nt: str):
    candidates = []
    fwd = nt
    rev = str(Seq(nt).reverse_complement())

    for strand_name, seq in [("forward", fwd), ("reverse_complement", rev)]:
        for frame in [0, 1, 2]:
            aa = translate_frame(seq, frame)
            pre_stop = aa.split("*")[0]
            internal_stops = aa.count("*")
            candidates.append({
                "strand": strand_name,
                "frame": frame + 1,
                "aa": aa,
                "aa_len_total": len(aa),
                "aa_len_before_first_stop": len(pre_stop),
                "stop_count": internal_stops,
            })

    best = sorted(
        candidates,
        key=lambda x: (x["aa_len_before_first_stop"], -x["stop_count"], x["aa_len_total"]),
        reverse=True
    )[0]
    return best, candidates

if st.button("Analyze Sequence"):
    if not sequence.strip():
        st.error("Please paste a sequence.")
        st.stop()

    nt = clean_sequence(sequence) blast_url = build_blast_url(nt)  # หรือ clean_seq ถ้าคุณใช้ชื่อนั้น

st.markdown("### 🔗 External Analysis")
st.markdown(f"[👉 Run NCBI BLAST (blastn)]({blast_url})")

    def build_blast_url(sequence: str) -> str:
    base_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    params = {
        "PROGRAM": "blastn",
        "PAGE_TYPE": "BlastSearch",
        "QUERY": sequence
    }
    return base_url + "?" + urllib.parse.urlencode(params)

    if len(nt) == 0:
        st.error("No valid A/T/G/C bases found after cleaning. Please check input.")
        st.stop()

    length = len(nt)
    gc = gc_content(nt)

    best, all_frames = best_orf_6frames(nt)
    early_stop_flag = best["aa_len_before_first_stop"] < 80
    many_stops_flag = best["stop_count"] > 1

    st.subheader("🔍 Analysis Result")
    st.write(f"Sample ID: {sample_id if sample_id else '-'}")
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
        st.success("ORF check: ✅ PASS (best frame shows no early stop issues).")

    with st.expander("Show translated AA (best frame)"):
        st.code(best["aa"], language="text")

    st.markdown("### Gene/Subtype Identification")
    st.info("Phase 1: ORF/QC sanity checks only. (Next step can add BLAST parsing.)")

    st.markdown("### QC Assessment")
    st.write("Status: **Screening complete.**")

