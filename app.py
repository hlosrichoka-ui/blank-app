import re
import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="HA/NA Sequence Identification Quick-Check", layout="centered")
st.title("HA/NA Sequence Identification Quick-Check")

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
    """
    Keep length stable:
    - remove FASTA headers
    - keep letters only
    - convert non-ATGC to N (do NOT delete)
    """
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    s = "".join(lines).upper()
    s = re.sub(r"[^A-Z]", "", s)
    s = re.sub(r"[^ATGC]", "N", s)
    return s

def best_orf_6frames(nt: str):
    """Simple ORF sanity check: longest AA before first stop across 6 frames."""
    fwd = nt
    rev = str(Seq(nt).reverse_complement())
    best_len = 0
    best_stop = 0
    for seq in (fwd, rev):
        for frame in (0, 1, 2):
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

def ranges_to_text(ranges):
    if not ranges:
        return "-"
    return ", ".join([f"{s}-{e}" for s, e in ranges])

def annotate_sequence(query: str, mismatch_positions: list, uncovered_positions: list, insertion_positions: list, width: int = 60) -> str:
    """
    UPPERCASE=match
    lowercase=mismatch (aligned but different)
    i=insertion (query aligned to gap in ref)
    N=uncovered (not aligned at all)
    """
    mis = set(mismatch_positions)
    unc = set(uncovered_positions)
    ins = set(insertion_positions)
    out = []
    for i, base in enumerate(query, start=1):
        if i in unc:
            out.append("N")
        elif i in ins:
            out.append("i")
        elif i in mis:
            out.append(base.lower())
        else:
            out.append(base)
    return "\n".join(["".join(out[i:i+width]) for i in range(0, len(out), width)])

def query_centric_local_compare(ref: str, qry: str):
    """
    LOCAL alignment (no need to start at base 1)

    %Identification (per your definition) = matches / input_length * 100
    Identity (aligned region)             = matches / aligned_query_bases * 100
    Coverage (of input aligned)           = aligned_query_bases / input_length * 100

    We compute everything using aln.aligned (no aln.format).
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    total_q = len(qry)
    if total_q == 0:
        return 0.0, 0.0, 0.0, [], [], [], 0, 0, [], []

    aln = next(iter(aligner.align(ref, qry)))
    ref_blocks = aln.aligned[0]
    qry_blocks = aln.aligned[1]

    if ref_blocks is None or qry_blocks is None or len(qry_blocks) == 0:
        uncovered = list(range(1, total_q + 1))
        return 0.0, 0.0, 0.0, [], [], uncovered, 0, 0, [], []

    # report ranges (1-based inclusive; end is inclusive in display)
    ref_ranges = [(int(rs) + 1, int(re)) for rs, re in ref_blocks]
    qry_ranges = [(int(qs) + 1, int(qe)) for qs, qe in qry_blocks]

    matches = 0
    mismatches = []
    insertions = []
    covered = set()

    aligned_query_bases = 0  # includes matches+mismatches+insertions (but not deletions)

    prev_rs = None
    prev_re = None
    prev_qs = None
    prev_qe = None

    for (rs, re), (qs, qe) in zip(ref_blocks, qry_blocks):
        rs, re, qs, qe = int(rs), int(re), int(qs), int(qe)

        # gaps between blocks => insertion or deletion
        if prev_rs is not None:
            r_gap = rs - prev_re
            q_gap = qs - prev_qe

            # insertion in query: q advances but ref doesn't
            if q_gap > 0 and r_gap == 0:
                for p in range(prev_qe + 1, qs + 1):  # 0-based -> include those bases
                    qpos = p  # 1-based later
                    insertions.append(qpos)
                    covered.add(qpos)
                aligned_query_bases += q_gap

            # deletion in query: ref advances but query doesn't (no query positions to mark)
            # if both gaps >0, treat query gap part as uncovered (rare); we ignore here.

        # aligned block length
        block_len = min(re - rs, qe - qs)
        # compare base-by-base
        for i in range(block_len):
            qpos_1based = (qs + i) + 1
            rbase = ref[rs + i]
            qbase = qry[qs + i]
            covered.add(qpos_1based)
            aligned_query_bases += 1
            if rbase == qbase:
                matches += 1
            else:
                mismatches.append(qpos_1based)

        prev_rs, prev_re, prev_qs, prev_qe = rs, re, qs, qe

    uncovered = [p for p in range(1, total_q + 1) if p not in covered]

    identification_pct = (matches / total_q) * 100.0
    identity_aligned = (matches / aligned_query_bases) * 100.0 if aligned_query_bases else 0.0
    coverage_input = (aligned_query_bases / total_q) * 100.0

    return (
        identification_pct,
        identity_aligned,
        coverage_input,
        mismatches,
        insertions,
        uncovered,
        matches,
        aligned_query_bases,
        ref_ranges,
        qry_ranges,
    )

# =========================================================
# UI (inputs)
# =========================================================
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)
query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

IDENTIFICATION_PASS = st.number_input("%Identification PASS threshold (matches/input length)", value=95.0, step=0.5)
COVERAGE_PASS = st.number_input("Coverage PASS threshold (% of input aligned)", value=90.0, step=1.0)

show_positions = st.checkbox("Show mismatch/insertion/uncovered positions", value=True)
show_annotated = st.checkbox("Show annotated input sequence", value=True)
show_ranges = st.checkbox("Show aligned start/end positions (REF & INPUT)", value=True)

# =========================================================
# Analyze
# =========================================================
if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q:
        st.error("Please paste a valid nucleotide sequence.")
        st.stop()

    HA_REF = clean_nt(HA_REF_RAW)
    NA_REF = clean_nt(NA_REF_RAW)

    ha = query_centric_local_compare(HA_REF, Q)
    na = query_centric_local_compare(NA_REF, Q)

    # Choose best gene by (%Identification of input, then Coverage)
    if (ha[0], ha[2]) >= (na[0], na[2]):
        gene_identified = "HA"
        gene_label = HA_REF_LABEL
        (ident_pct, ident_aligned, cov_pct,
         mismatch_pos, insertion_pos, uncovered_pos,
         matches, aligned_bases, ref_ranges, qry_ranges) = ha
    else:
        gene_identified = "NA"
        gene_label = NA_REF_LABEL
        (ident_pct, ident_aligned, cov_pct,
         mismatch_pos, insertion_pos, uncovered_pos,
         matches, aligned_bases, ref_ranges, qry_ranges) = na

    # ORF check
    best_len, stop_count = best_orf_6frames(Q)
    orf_status = "PASS" if (best_len >= 80 and stop_count <= 1) else "INVESTIGATE"

    # QC Assessment
    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
        qc_flag = "FAIL"
    elif ident_pct < IDENTIFICATION_PASS or cov_pct < COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low %Identification/Coverage vs thresholds)"
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

    st.write(f"**%Identification (matches / input length):** {ident_pct:.2f}")
    st.caption("= (จำนวนเบสที่เหมือนกัน / จำนวนเบสของ input ทั้งหมด) × 100")

    st.write(f"**Identity (% of aligned region):** {ident_aligned:.2f}")
    st.caption("= ตรงกันกี่ % เฉพาะส่วนที่ align ได้ (BLAST-like)")

    st.write(f"**Coverage (% of input aligned):** {cov_pct:.2f}")
    st.write(f"**ORF Check:** {orf_status}")
    st.markdown(f"### QC Assessment: {qc_assessment}")

    if show_ranges:
        with st.expander("Aligned start/end positions (LOCAL alignment; not necessarily from first base)"):
            st.write("**INPUT (query) aligned ranges (1-based):**", ranges_to_text(qry_ranges))
            st.write("**REF aligned ranges (1-based):**", ranges_to_text(ref_ranges))

    with st.expander("Show query-centric breakdown (traceability)"):
        st.write(f"Input length: **{len(Q)} bp**")
        st.write(f"Aligned input bases (coverage numerator): **{aligned_bases} / {len(Q)}**")
        st.write(f"Matched bases (numerator for %Identification): **{matches} / {len(Q)}**")
        st.write(f"Mismatch count: **{len(mismatch_pos)}**")
        st.write(f"Insertion count: **{len(insertion_pos)}**")
        st.write(f"Uncovered count: **{len(uncovered_pos)}**")

    if show_positions:
        with st.expander("Mismatch / Insertion / Uncovered positions (1-based on INPUT)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Mismatch")
                st.write(f"Total: **{len(mismatch_pos)}**")
                st.code(", ".join(map(str, mismatch_pos[:300])) + ("" if len(mismatch_pos) <= 300 else f"\n... (+{len(mismatch_pos)-300} more)"))
            with col2:
                st.markdown("#### Insertion (ref gap)")
                st.write(f"Total: **{len(insertion_pos)}**")
                st.code(", ".join(map(str, insertion_pos[:300])) + ("" if len(insertion_pos) <= 300 else f"\n... (+{len(insertion_pos)-300} more)"))
            with col3:
                st.markdown("#### Uncovered")
                st.write(f"Total: **{len(uncovered_pos)}**")
                st.code(", ".join(map(str, uncovered_pos[:300])) + ("" if len(uncovered_pos) <= 300 else f"\n... (+{len(uncovered_pos)-300} more)"))

    if show_annotated:
        with st.expander("Annotated input sequence (UPPERCASE=match, lowercase=mismatch, i=insertion, N=uncovered)"):
            st.code(annotate_sequence(Q, mismatch_pos, uncovered_pos, insertion_pos), language="text")

    # =====================================================
    # Download QC report (CSV)
    # =====================================================
    report = pd.DataFrame([{
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene_identified,
        "Subtype": SUBTYPE_LABEL,
        "%Identification (matches/input)": round(ident_pct, 2),
        "Identity (% of aligned region)": round(ident_aligned, 2),
        "Coverage (% of input aligned)": round(cov_pct, 2),
        "Input length (bp)": len(Q),
        "Aligned input bases": aligned_bases,
        "Matched bases": matches,
        "Mismatch count": len(mismatch_pos),
        "Insertion count": len(insertion_pos),
        "Uncovered count": len(uncovered_pos),
        "INPUT aligned ranges (1-based)": ranges_to_text(qry_ranges),
        "REF aligned ranges (1-based)": ranges_to_text(ref_ranges),
        "Mismatch positions (1-based input)": ";".join(map(str, mismatch_pos)) if mismatch_pos else "",
        "Insertion positions (1-based input)": ";".join(map(str, insertion_pos)) if insertion_pos else "",
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
        file_name=f"{sample_id or 'qc_report'}_HA_NA_identification.csv",
        mime="text/csv"
    )

st.caption(
    "LOCAL alignment; %Identification = matches / input length × 100 (input ไม่ต้องเริ่มตรงกับ ref ที่เบสแรก)"
)
