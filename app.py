import re
import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="HA/NA Sequence Identity Quick-Check", layout="centered")
st.title("HA/NA Sequence Identity Quick-Check")

# =========================================================
# Reference information
# =========================================================
ORGANISM_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9))"
SUBTYPE_LABEL = "H7N9"

HA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 4 hemagglutinin (HA) gene"
NA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 6 neuraminidase (NA) gene"

# =========================================================
# Reference sequences (user-provided)
# =========================================================
HA_REF_RAW = """ATGAACACTCAAATCCTGGTATTCGCTCTGATTGCGATCATTCCAACAAATGCAGACAAAATCTGCCTCG
GACATCATGCCGTGTCAAACGGAACCAAAGTAAACACATTAACTGAAAGAGGAGTGGAAGTCGTCAATGC
AACTGAAACAGTGGAACGAACAAACATCCCCAGGATCTGCTCAAAAGGGAAAAGGACAGTTGACCTCGGT
CAATGTGGACTCCTGGGGACAATCACTGGACCACCTCAATGTGACCAATTCCTAGAATTTTCAGCCGATT
TAATTATTGAGAGGCGAGAAGGAAGTGATGTCTGTTATCCTGGGAAATTCGTGAATGAAGAAGCTCTGAG
GCAAATTCTCAGAGAATCAGGCGGAATTGACAAGGAAGCAATGGGATTCACATACAGTGGAATAAGAACT
AATGGAGCAACCAGTGCATGTAGGAGATCAGGATCTTCATTCTATGCAGAAATGAAATGGCTCCTGTCAA
ACACAGATAATGCTGCATTCCCGCAGATGACTAAGTCATATAAAAATACAAGAAAAAGCCCAGCTCTAAT
AGTATGGGGGATCCATCATTCCGTATCAACTGCAGAGCAAACCAAGCTATATGGGAGTGGAAACAAACTG
GTGACAGTTGGGAGTTCTAATTATCAACAATCTTTTGTACCGAGTCCAGGAGCGAGACCACAAGTTAATG
GTCTATCTGGAAGAATTGACTTTCATTGGCTAATGCTAAATCCCAATGATACAGTCACTTTCAGTTTCAA
TGGGGCTTTCATAGCTCCAGACCGTGCAAGCTTCCTGAGAGGAAAATCTATGGGAATCCAGAGTGGAGTA
CAGGTTGATGCCAATTGTGAAGGGGACTGCTATCATAGTGGAGGGACAATAATAAGTAACTTGCCATTTC
AGAACATAGATAGCAGGGCAGTTGGAAAATGTCCGAGATATGTTAAGCAAAGGAGTCTGCTGCTAGCAAC
AGGGATGAAGAATGTTCCTGAGATTCCAAAAGGAAGAGGCCTATTTGGTGCTATAGCGGGTTTCATTGAA
AATGGATGGGAAGGCCTAATTGATGGTTGGTATGGTTTCAGACACCAGAATGCACAGGGAGAGGGAACTG
CTGCAGATTACAAAAGCACTCAATCGGCAATTGATCAAATAACAGGAAAATTAAACCGGCTTATAGAAAA
AACCAACCAACAATTTGAGTTGATAGACAATGAATTCAATGAGGTAGAGAAGCAAATCGGTAATGTGATA
AATTGGACCAGAGATTCTATAACAGAAGTGTGGTCATACAATGCTGAACTCTTGGTAGCAATGGAGAACC
AGCATACAATTGATCTGGCTGATTCAGAAATGGACAAACTGTACGAACGAGTGAAAAGACAGCTGAGAGA
GAATGCTGAAGAAGATGGCACTGGTTGCTTTGAAATATTTCACAAGTGTGATGATGACTGTATGGCCAGT
ATTAGAAATAACACCTATGATCACAGCAAATACAGGGAAGAGGCAATGCAAAATAGAATACAGATTGACC
CAGTCAAACTAAGCAGCGGCTACAAAGATGTGATACTTTGGTTTAGCTTCGGGGCATCATGTTTCATACT
TCTAGCCATTGTAATGGGCCTTGTCTTCATATGTGTAAAGAATGGAAACATGCGGTGCACTATTTGTATA
TAA"""

NA_REF_RAW = """ATGAATCCAAATCAGAAGATTCTATGCACTTCAGCCACTGCTATCATAATAGGCGCAATCGCAGTACTCA
TTGGAATGGCAAACCTAGGATTGAACATAGGACTGCATCTAAAACCGGGCTGCAATTGCTCACACTCACA
ACCTGAAACAACCAACACAAGCCAAACAATAATAAACAACTATTATAATGAAACAAACATCACCAAYATC
CAAATGGAAGAGAGAACAAGCAGGAATTTCAATAACTTAACTAAAGGGCTCTGTACTATAAATTCATGGC
ACATATATGGGAAAGACAATGCAGTAAGAATTGGAGAGAGCTCGGATGTTTTAGTCACAAGAGAACCCTA
TGTTTCATGCGACCCAGATGAATGCAGGTTCTATGCTCTCAGCCAAGGAACAACAATCAGAGGGAAACAC
TCAAACGGAACAATACACGATAGGTCCCAGTATCGCGCCCTGATAAGCTGGCCACTATCATCACCGCCCA
CAGTGTACAACAGCAGGGTGGAATGCATTGGGTGGTCAAGTACTAGTTGCCATGATGGCAAATCCAGGAT
GTCAATATGTATATCAGGACCAAACAACAATGCATCTGCAGTAGTATGGTACAACAGAAGGCCTGTTGCA
GAAATTAACACATGGGCCCGAAACATACTAAGAACACAGGAATCTGAATGTGTATGCCACAACGGCGTAT
GCCCAGTAGTGTTCACCGATGGGTCTGCCACTGGACCTGCAGACACAAGAATATACTATTTTAAAGAGGG
GAAAATATTGAAATGGGAGTCTCTGACTGGAACTGCTAAGCATATTGAAGAATGCTCATGTTACGGGGAA
CGAACAGGAATTACCTGCACATGCAGGGACAATTGGCAGGGCTCAAATAGACCAGTGATTCAGATAGACC
CAGTAGCAATGACACACACTAGTCAATATATATGCAGTCCTGTTCTTACAGACAATCCCCGACCGAATGA
CCCAAATATAGGTAAGTGTAATGACCCTTATCCAGGTAATAATAACAATGGAGTCAAGGGATTCTCATAC
CTGGATGGGGCTAACACTTGGCTAGGGAGGACAATAAGCACAGCCTCGAGGTCTGGATACGAGATGTTAA
AAGTGCCAAATGCATTGACAGATGATAGATCAAAGCCCATTCAAGGTCAGACAATTGTATTAAACGCTGA
CTGGAGTGGTTACAGTGGATCTTTCATGGACTATTGGGCTGAAGGGGACTGCTATCGAGCGTGTTTTTAT
GTGGAGTTGATACGTGGAAGACCCAAGGAGGATAAAGTGTGGTGGACCAGCAATAGTATAGTATCGATGT
GTTCCAGTACAGAATTCCTGGGACAATGGAACTGGCCTGATGGGGCTAAAATAGAGTACTTCCTCTAA"""

# =========================================================
# Helper functions
# =========================================================
def clean_nt(seq: str) -> str:
    """Remove FASTA headers and keep only A/T/G/C. Drops ambiguity codes (e.g., Y)."""
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

def best_orf_6frames(nt: str):
    """Simple ORF sanity check: longest AA before first stop across 6 frames."""
    fwd = nt
    rev = str(Seq(nt).reverse_complement())
    best_len = 0
    best_stop = 0
    for seq in [fwd, rev]:
        for frame in [0, 1, 2]:
            trimmed = seq[frame:]
            trimmed = trimmed[: (len(trimmed) // 3) * 3]
            if not trimmed:
                continue
            aa = str(Seq(trimmed).translate(to_stop=False))
            pre = aa.split("*")[0]
            if len(pre) > best_len:
                best_len = len(pre)
                best_stop = aa.count("*")
    return best_len, best_stop

def query_centric_local_compare(ref: str, qry: str):
    """
    LOCAL alignment: finds best matching region (does NOT need to start at first base).

    QUERY-centric metrics:
      Identity (% of input)        = matches / len(query) * 100
      Identity (% aligned region)  = matches / aligned_query_bases * 100
      Coverage (% of input aligned)= aligned_query_bases / len(query) * 100

    Also returns:
      mismatch_positions_abs (1-based on query)
      uncovered_positions_abs (1-based on query)
      ref_ranges (1-based inclusive): list of (start,end) for aligned blocks on REF
      qry_ranges (1-based inclusive): list of (start,end) for aligned blocks on QUERY
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"  # ✅ important: local alignment
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(ref, qry)))

    ref_blocks = aln.aligned[0]
    qry_blocks = aln.aligned[1]

    total_query = len(qry)

    # Robust empty-check across Biopython/NumPy behaviors
    if qry_blocks is None or len(qry_blocks) == 0:
        uncovered = list(range(1, total_query + 1))
        return 0.0, 0.0, 0.0, [], uncovered, 0, 0, [], []

    # 1-based inclusive ranges for reporting
    ref_ranges = [(int(rs) + 1, int(re)) for rs, re in ref_blocks]
    qry_ranges = [(int(qs) + 1, int(qe)) for qs, qe in qry_blocks]

    # Covered query positions (1-based)
    covered = set()
    for qs, qe in qry_blocks:
        for p in range(int(qs) + 1, int(qe) + 1):
            covered.add(p)

    uncovered = [p for p in range(1, total_query + 1) if p not in covered]

    # Alignment strings for mismatch detection
    lines = aln.format().splitlines()
    ref_aln = lines[0].replace(" ", "")
    qry_aln = lines[2].replace(" ", "")

    covered_sorted = sorted(covered)
    idx = 0

    matches = 0
    aligned_query_bases = 0
    mismatch_abs = []

    for r, q in zip(ref_aln, qry_aln):
        if q == "-":
            continue
        if idx >= len(covered_sorted):
            break
        q_abs = covered_sorted[idx]
        idx += 1

        aligned_query_bases += 1
        if r == q:
            matches += 1
        else:
            mismatch_abs.append(q_abs)

    identity_input = (matches / total_query * 100) if total_query else 0.0
    identity_aligned = (matches / aligned_query_bases * 100) if aligned_query_bases else 0.0
    coverage_input = (aligned_query_bases / total_query * 100) if total_query else 0.0

    return (
        identity_input,
        identity_aligned,
        coverage_input,
        mismatch_abs,
        uncovered,
        matches,
        aligned_query_bases,
        ref_ranges,
        qry_ranges,
    )

def annotate_sequence(query: str, mismatch_positions: list, uncovered_positions: list, width: int = 60) -> str:
    """
    Annotated sequence:
      - UPPERCASE = match
      - lowercase = mismatch (aligned but different)
      - N = uncovered (not aligned)
    """
    mis = set(mismatch_positions)
    unc = set(uncovered_positions)
    out = []
    for i, base in enumerate(query, start=1):
        if i in unc:
            out.append("N")
        elif i in mis:
            out.append(base.lower())
        else:
            out.append(base)
    blocks = ["".join(out[i:i+width]) for i in range(0, len(out), width)]
    return "\n".join(blocks)

def ranges_to_text(ranges):
    """Pretty print list of (start,end) ranges."""
    if not ranges:
        return "-"
    return ", ".join([f"{s}-{e}" for s, e in ranges])

# =========================================================
# UI (inputs)
# =========================================================
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)
query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

# QC thresholds (adjust to your SOP)
IDENTITY_PASS = st.number_input("Identity PASS threshold (% of input)", value=95.0, step=0.5)
COVERAGE_PASS = st.number_input("Coverage PASS threshold (% of input aligned)", value=90.0, step=1.0)

show_positions = st.checkbox("Show mismatch/uncovered positions", value=True)
show_annotated = st.checkbox("Show annotated input sequence", value=True)
show_ranges = st.checkbox("Show aligned start/end positions (REF & INPUT)", value=True)

# =========================================================
# Analyze
# =========================================================
if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q:
        st.error("Please paste a valid nucleotide sequence (A/T/G/C).")
        st.stop()

    HA_REF = clean_nt(HA_REF_RAW)
    NA_REF = clean_nt(NA_REF_RAW)

    ha = query_centric_local_compare(HA_REF, Q)
    na = query_centric_local_compare(NA_REF, Q)

    # Choose best gene by (Identity % of input, then Coverage % of input)
    if (ha[0], ha[2]) >= (na[0], na[2]):
        gene_identified = "HA"
        gene_label = HA_REF_LABEL
        identity_input, identity_aligned, coverage_input, mismatch_pos, uncovered_pos, matches, aligned_bases, ref_ranges, qry_ranges = ha
    else:
        gene_identified = "NA"
        gene_label = NA_REF_LABEL
        identity_input, identity_aligned, coverage_input, mismatch_pos, uncovered_pos, matches, aligned_bases, ref_ranges, qry_ranges = na

    # ORF check
    best_len, stop_count = best_orf_6frames(Q)
    orf_status = "PASS" if (best_len >= 80 and stop_count <= 1) else "INVESTIGATE"

    # QC Assessment
    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
        qc_flag = "FAIL"
    elif identity_input < IDENTITY_PASS or coverage_input < COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low identity/coverage vs thresholds)"
        qc_flag = "INVESTIGATE"
    elif orf_status != "PASS":
        qc_assessment = "⚠️ INVESTIGATE (ORF check)"
        qc_flag = "INVESTIGATE"
    else:
        qc_assessment = "✅ PASS"
        qc_flag = "PASS"

    # =====================================================
    # Display results
    # =====================================================
    st.markdown("---")
    st.subheader("🔍 Analysis Result")
    st.write(f"**Sample ID:** {sample_id if sample_id else '-'}")
    st.write(f"**Organism:** {ORGANISM_LABEL}")
    st.write(f"**Gene Identified:** {gene_identified} — ({gene_label})")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")

    st.write(f"**Identity (% of input):** {identity_input:.2f}")
    st.caption("= sample ตรงกับ reference กี่ % ของ input ทั้งหมด (นับส่วนที่ไม่ align เป็น non-match)")

    st.write(f"**Identity (% of aligned region):** {identity_aligned:.2f}")
    st.caption("= ตรงกันกี่ % เฉพาะส่วนที่ align ได้ (BLAST-like)")

    st.write(f"**Coverage (% of input aligned):** {coverage_input:.2f}")
    st.write(f"**ORF Check:** {orf_status}")

    st.markdown(f"### QC Assessment: {qc_assessment}")

    # =====================================================
    # Alignment position ranges (start/end)
    # =====================================================
    if show_ranges:
        with st.expander("Aligned start/end positions (LOCAL alignment; not necessarily from first base)"):
            st.write("**INPUT (query) aligned ranges (1-based):**", ranges_to_text(qry_ranges))
            st.write("**REF aligned ranges (1-based):**", ranges_to_text(ref_ranges))
            st.caption("Note: local alignment finds the best matching region(s); alignment does not need to start at position 1.")

    # =====================================================
    # QC traceability breakdown
    # =====================================================
    with st.expander("Show query-centric breakdown (traceability)"):
        st.write(f"Input length: **{len(Q)} bp**")
        st.write(f"Aligned input bases (coverage numerator): **{aligned_bases} / {len(Q)}**")
        st.write(f"Matched bases (counted toward Identity % of input): **{matches} / {len(Q)}**")
        st.write(f"Mismatch count (aligned but different): **{len(mismatch_pos)}**")
        st.write(f"Uncovered count (not aligned to ref): **{len(uncovered_pos)}**")

    # =====================================================
    # Highlight mismatch / uncovered positions + annotated sequence
    # =====================================================
    if show_positions:
        with st.expander("Mismatch / Uncovered positions (1-based on INPUT)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Mismatch positions (aligned but not equal)")
                if mismatch_pos:
                    st.write(f"Total mismatches: **{len(mismatch_pos)}**")
                    st.code(", ".join(map(str, mismatch_pos)))
                else:
                    st.write("None")

            with col2:
                st.markdown("#### Uncovered positions (no alignment to ref)")
                if uncovered_pos:
                    st.write(f"Total uncovered: **{len(uncovered_pos)}**")
                    preview = uncovered_pos[:200]
                    tail_note = "" if len(uncovered_pos) <= 200 else f"\n... (+{len(uncovered_pos)-200} more)"
                    st.code(", ".join(map(str, preview)) + tail_note)
                else:
                    st.write("None")

    if show_annotated:
        with st.expander("Annotated input sequence (UPPERCASE=match, lowercase=mismatch, N=uncovered)"):
            st.code(annotate_sequence(Q, mismatch_pos, uncovered_pos), language="text")

    # =====================================================
    # Download QC report (CSV)
    # =====================================================
    report = pd.DataFrame([{
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene_identified,
        "Subtype": SUBTYPE_LABEL,
        "Identity (% of input)": round(identity_input, 2),
        "Identity (% of aligned region)": round(identity_aligned, 2),
        "Coverage (% of input aligned)": round(coverage_input, 2),
        "Input length (bp)": len(Q),
        "Aligned input bases": aligned_bases,
        "Matched bases": matches,
        "Mismatch count": len(mismatch_pos),
        "Uncovered count": len(uncovered_pos),
        "INPUT aligned ranges (1-based)": ranges_to_text(qry_ranges),
        "REF aligned ranges (1-based)": ranges_to_text(ref_ranges),
        "Mismatch positions (1-based input)": ";".join(map(str, mismatch_pos)) if mismatch_pos else "",
        "Uncovered positions (1-based input)": ";".join(map(str, uncovered_pos)) if uncovered_pos else "",
        "ORF Check": orf_status,
        "QC Flag": qc_flag,
        "QC Assessment": qc_assessment,
        "Reference used": gene_label,
        "Alignment mode": "local",
    }])

    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download QC Report (CSV)",
        data=csv_bytes,
        file_name=f"{sample_id or 'qc_report'}_HA_NA_identity.csv",
        mime="text/csv"
    )

st.caption(
    "This app uses LOCAL alignment to find the best matching region(s); it does not require matching from the first base."
)
