import re
import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="HA/NA Sequence Identity Quick-Check", layout="centered")
st.title("HA/NA Sequence Identity Quick-Check (BLAST-like)")

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
# Helpers
# =========================================================
def clean_nt(seq: str) -> str:
    """Remove FASTA headers and keep only A/T/G/C. Drops ambiguity codes."""
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

def reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())

def best_orf_6frames(nt: str):
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

def ranges_to_text(ranges):
    if not ranges:
        return "-"
    return ", ".join([f"{s}-{e}" for s, e in ranges])

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

def local_align_blastlike(ref: str, qry: str):
    """
    LOCAL alignment (BLAST-like HSP behavior).
    Returns BOTH:
      - Query-centric metrics (identity over input, coverage over input)
      - BLAST-like metrics (Identities X/Y, Gaps g/Y, %Identity over alignment length)

    Output tuple:
      identity_input_pct,
      identity_aligned_pct (matches / aligned query bases),
      coverage_input_pct,
      mismatch_positions_abs (1-based on query),
      uncovered_positions_abs (1-based on query),
      matches (X),
      aligned_query_bases,
      ref_ranges (1-based inclusive),
      qry_ranges (1-based inclusive),
      aln_length (Y; includes gaps),
      gaps_total (g; '-' in either aligned string),
      identity_alnlen_pct (X/Y * 100),
      gaps_pct (g/Y * 100)
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(ref, qry)))

    ref_blocks = aln.aligned[0]
    qry_blocks = aln.aligned[1]
    total_query = len(qry)

    if qry_blocks is None or len(qry_blocks) == 0:
        uncovered = list(range(1, total_query + 1))
        return (0.0, 0.0, 0.0, [], uncovered, 0, 0, [], [], 0, 0, 0.0, 0.0)

    # 1-based inclusive ranges for reporting
    ref_ranges = [(int(rs) + 1, int(re)) for rs, re in ref_blocks]
    qry_ranges = [(int(qs) + 1, int(qe)) for qs, qe in qry_blocks]

    # Covered query positions (1-based)
    covered = set()
    for qs, qe in qry_blocks:
        for p in range(int(qs) + 1, int(qe) + 1):
            covered.add(p)
    uncovered = [p for p in range(1, total_query + 1) if p not in covered]

    # Alignment strings
    lines = aln.format().splitlines()
    ref_aln = lines[0].replace(" ", "")
    qry_aln = lines[2].replace(" ", "")

    aln_length = len(ref_aln)  # includes gaps (BLAST "Length" for HSP)
    gaps_total = ref_aln.count("-") + qry_aln.count("-")
    gaps_pct = (gaps_total / aln_length * 100) if aln_length else 0.0

    # Map mismatch positions to absolute query coordinates (1-based)
    covered_sorted = sorted(covered)
    idx = 0

    matches = 0  # identities X
    aligned_query_bases = 0  # query letters excluding gaps
    mismatch_abs = []

    for r, q in zip(ref_aln, qry_aln):
        if q == "-":
            continue
        if idx >= len(covered_sorted):
            break
        q_abs = covered_sorted[idx]
        idx += 1

        aligned_query_bases += 1
        if r == q and r != "-":
            matches += 1
        else:
            mismatch_abs.append(q_abs)

    identity_input_pct = (matches / total_query * 100) if total_query else 0.0
    identity_aligned_pct = (matches / aligned_query_bases * 100) if aligned_query_bases else 0.0
    coverage_input_pct = (aligned_query_bases / total_query * 100) if total_query else 0.0

    identity_alnlen_pct = (matches / aln_length * 100) if aln_length else 0.0

    return (
        identity_input_pct,
        identity_aligned_pct,
        coverage_input_pct,
        mismatch_abs,
        uncovered,
        matches,
        aligned_query_bases,
        ref_ranges,
        qry_ranges,
        aln_length,
        gaps_total,
        identity_alnlen_pct,
        gaps_pct,
    )

def pick_best(res_a, res_b):
    """
    Pick best result BLAST-like:
      1) highest identity over alignment length (X/Y)
      2) then longer alignment length (Y)
    """
    # res[11]=identity_alnlen_pct, res[9]=aln_length
    return res_a if (res_a[11], res_a[9]) >= (res_b[11], res_b[9]) else res_b

# =========================================================
# UI
# =========================================================
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)
query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

# BLAST-like thresholds (you can tighten like 99.0)
IDENTITY_ALNLEN_PASS = st.number_input("BLAST-like Identity PASS threshold (% of alignment length)", value=99.0, step=0.1)
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

    # Evaluate BOTH strands (sense + antisense) to support A-T / C-G pairing concept
    Q_rc = reverse_complement(Q)

    # HA
    ha_sense = local_align_blastlike(HA_REF, Q)
    ha_anti = local_align_blastlike(HA_REF, Q_rc)
    ha_best = pick_best(ha_sense, ha_anti)
    ha_strand = "antisense (reverse-complement)" if ha_best == ha_anti else "sense"

    # NA
    na_sense = local_align_blastlike(NA_REF, Q)
    na_anti = local_align_blastlike(NA_REF, Q_rc)
    na_best = pick_best(na_sense, na_anti)
    na_strand = "antisense (reverse-complement)" if na_best == na_anti else "sense"

    # Choose HA vs NA (BLAST-like)
    if (ha_best[11], ha_best[9]) >= (na_best[11], na_best[9]):
        gene_identified = "HA"
        gene_label = HA_REF_LABEL
        strand_used = ha_strand
        res = ha_best
    else:
        gene_identified = "NA"
        gene_label = NA_REF_LABEL
        strand_used = na_strand
        res = na_best

    (
        identity_input_pct,
        identity_aligned_pct,
        coverage_input_pct,
        mismatch_pos,
        uncovered_pos,
        matches_X,
        aligned_query_bases,
        ref_ranges,
        qry_ranges,
        aln_length_Y,
        gaps_g,
        identity_alnlen_pct,
        gaps_pct,
    ) = res

    # ORF check (on original cleaned input strand)
    best_len, stop_count = best_orf_6frames(Q)
    orf_status = "PASS" if (best_len >= 80 and stop_count <= 1) else "INVESTIGATE"

    # QC Assessment (BLAST-like)
    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
        qc_flag = "FAIL"
    elif identity_alnlen_pct < IDENTITY_ALNLEN_PASS or coverage_input_pct < COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low BLAST-like identity / coverage)"
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
    st.subheader("🔍 Analysis Result (BLAST-like)")
    st.write(f"**Sample ID:** {sample_id if sample_id else '-'}")
    st.write(f"**Organism:** {ORGANISM_LABEL}")
    st.write(f"**Gene Identified:** {gene_identified} — ({gene_label})")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")
    st.write(f"**Strand used:** {strand_used}")

    # BLAST-like lines
    st.write(f"**Identities:** {matches_X}/{aln_length_Y} ({identity_alnlen_pct:.2f}%)")
    st.write(f"**Gaps:** {gaps_g}/{aln_length_Y} ({gaps_pct:.2f}%)")
    st.write(f"**Alignment length (bp):** {aln_length_Y}")

    # Also show QC-friendly query-centric summaries
    st.markdown("#### QC-friendly (query-centric) summaries")
    st.write(f"**Identity (% of input):** {identity_input_pct:.2f}%")
    st.write(f"**Identity (% of aligned region; BLAST-like but excludes gap-columns):** {identity_aligned_pct:.2f}%")
    st.write(f"**Coverage (% of input aligned):** {coverage_input_pct:.2f}%")
    st.write(f"**ORF Check:** {orf_status}")

    st.markdown(f"### QC Assessment: {qc_assessment}")

    # =====================================================
    # Alignment ranges
    # =====================================================
    if show_ranges:
        with st.expander("Aligned start/end positions (LOCAL alignment; not necessarily from first base)"):
            st.write("**INPUT aligned ranges (1-based):**", ranges_to_text(qry_ranges))
            st.write("**REF aligned ranges (1-based):**", ranges_to_text(ref_ranges))

    # =====================================================
    # Positions and annotated sequence
    # =====================================================
    if show_positions:
        with st.expander("Mismatch / Uncovered positions (1-based on INPUT)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Mismatch positions (aligned but not equal)")
                st.write("None" if not mismatch_pos else f"Total: {len(mismatch_pos)}")
                if mismatch_pos:
                    st.code(", ".join(map(str, mismatch_pos)))
            with col2:
                st.markdown("#### Uncovered positions (no alignment to ref)")
                st.write("None" if not uncovered_pos else f"Total: {len(uncovered_pos)}")
                if uncovered_pos:
                    preview = uncovered_pos[:200]
                    tail = "" if len(uncovered_pos) <= 200 else f"\n... (+{len(uncovered_pos)-200} more)"
                    st.code(", ".join(map(str, preview)) + tail)

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
        "Strand used": strand_used,

        "Identities (X)": matches_X,
        "Alignment length (Y)": aln_length_Y,
        "BLAST-like Identity (% X/Y)": round(identity_alnlen_pct, 2),

        "Gaps (g)": gaps_g,
        "Gaps (% g/Y)": round(gaps_pct, 2),

        "Identity (% of input)": round(identity_input_pct, 2),
        "Identity (% of aligned region)": round(identity_aligned_pct, 2),
        "Coverage (% of input aligned)": round(coverage_input_pct, 2),

        "Aligned query bases (no gaps)": aligned_query_bases,

        "INPUT aligned ranges (1-based)": ranges_to_text(qry_ranges),
        "REF aligned ranges (1-based)": ranges_to_text(ref_ranges),

        "Mismatch count": len(mismatch_pos),
        "Uncovered count": len(uncovered_pos),
        "Mismatch positions (1-based input)": ";".join(map(str, mismatch_pos)) if mismatch_pos else "",
        "Uncovered positions (1-based input)": ";".join(map(str, uncovered_pos)) if uncovered_pos else "",

        "ORF Check": orf_status,
        "QC Flag": qc_flag,
        "QC Assessment": qc_assessment,
        "Reference used": gene_label,
        "Alignment mode": "local",
    }])

    st.download_button(
        label="Download QC Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_id or 'qc_report'}_HA_NA_blastlike.csv",
        mime="text/csv"
    )

st.caption(
    "BLAST-like behavior: LOCAL alignment reports the best matching region (HSP). "
    "Identities and Gaps are calculated over alignment length (including gap-columns), similar to NCBI BLAST."
)
