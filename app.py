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

# -------------------------
# Reference sequences (user-provided)
# -------------------------
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
    Local alignment to find best matching region in REF.
    Output metrics are QUERY-centric:
      - identity_over_query = matches / len(query) * 100  (unmatched query bases count as non-match)
      - coverage_over_query = aligned_query_bases / len(query) * 100
    Also returns:
      - mismatch_positions_abs (1-based positions on the ORIGINAL query)
      - uncovered_positions_abs (query bases not included in alignment at all)
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(ref, qry)))

    # blocks of aligned coordinates: list of (start,end) segments
    # aln.aligned[1] corresponds to query coordinates in original qry string
    q_blocks = aln.aligned[1]  # e.g., [(q0,q1), (q2,q3), ...]
    if len(q_blocks) == 0:
        # no alignment at all
        total = len(qry)
        return 0.0, 0.0, [], list(range(1, total + 1)), 0, 0

    # Reconstruct aligned strings (for mismatch detection within aligned part)
    lines = aln.format().splitlines()
    ref_aln = lines[0].replace(" ", "")
    qry_aln = lines[2].replace(" ", "")

    total_query = len(qry)

    # Build a set of query positions that are covered by alignment (absolute, 1-based)
    covered = set()
    for (qs, qe) in q_blocks:
        # qs, qe are 0-based half-open
        for p in range(qs + 1, qe + 1):
            covered.add(p)

    uncovered = [p for p in range(1, total_query + 1) if p not in covered]

    # To compute mismatches in aligned region with absolute positions:
    # We iterate through qry_aln characters and map them to absolute query positions.
    # But local alignment output doesn't show leading/trailing unaligned; we need the starting absolute pos.
    # We'll map by walking through alignment and consuming query coordinates using q_blocks.

    # Create an iterator over all covered query positions in increasing order
    covered_sorted = sorted(covered)
    covered_iter_idx = 0

    matches = 0
    aligned_query_bases = 0
    mismatch_abs = []

    for r, q in zip(ref_aln, qry_aln):
        if q == "-":
            continue  # gap in query doesn't consume query position
        # this consumes one query base => take next covered position
        if covered_iter_idx >= len(covered_sorted):
            # fallback safety
            break
        q_abs_pos = covered_sorted[covered_iter_idx]
        covered_iter_idx += 1

        aligned_query_bases += 1
        if r == q:
            matches += 1
        else:
            mismatch_abs.append(q_abs_pos)

    identity_over_query = (matches / total_query * 100) if total_query else 0.0
    coverage_over_query = (aligned_query_bases / total_query * 100) if total_query else 0.0

    return identity_over_query, coverage_over_query, mismatch_abs, uncovered, matches, aligned_query_bases


# =========================================================
# UI Inputs
# =========================================================
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)

query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

IDENTITY_PASS = st.number_input("Identity PASS threshold (% of input)", value=95.0, step=0.5)
COVERAGE_PASS = st.number_input("Coverage PASS threshold (% of input aligned)", value=90.0, step=1.0)

show_positions = st.checkbox("Show mismatch/uncovered positions", value=True)

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

    ha_id, ha_cov, ha_mis, ha_uncovered, ha_matches, ha_aligned = query_centric_local_compare(HA_REF, Q)
    na_id, na_cov, na_mis, na_uncovered, na_matches, na_aligned = query_centric_local_compare(NA_REF, Q)

    # Decide best gene by (identity, coverage)
    if (ha_id, ha_cov) >= (na_id, na_cov):
        gene_identified = "HA"
        best_identity, best_coverage = ha_id, ha_cov
        mismatch_pos, uncovered_pos = ha_mis, ha_uncovered
        matches, aligned_bases = ha_matches, ha_aligned
        gene_label = HA_REF_LABEL
    else:
        gene_identified = "NA"
        best_identity, best_coverage = na_id, na_cov
        mismatch_pos, uncovered_pos = na_mis, na_uncovered
        matches, aligned_bases = na_matches, na_aligned
        gene_label = NA_REF_LABEL

    # ORF check
    best_len, stop_count = best_orf_6frames(Q)
    orf_status = "PASS" if (best_len >= 80 and stop_count <= 1) else "INVESTIGATE"

    # Critical mutation placeholder
    critical_mutation = "None (not assessed in Phase 1)"

    # QC Assessment
    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
        qc_flag = "FAIL"
    elif best_identity < IDENTITY_PASS or best_coverage < COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low identity/coverage vs thresholds)"
        qc_flag = "INVESTIGATE"
    elif orf_status != "PASS":
        qc_assessment = "⚠️ INVESTIGATE (ORF check)"
        qc_flag = "INVESTIGATE"
    else:
        qc_assessment = "✅ PASS"
        qc_flag = "PASS"

    # =====================================================
    # Display result (as requested)
    # =====================================================
    st.markdown("---")
    st.subheader("🔍 Analysis Result")
    st.write(f"**Sample ID:** {sample_id if sample_id else '-'}")
    st.write(f"**Gene Identified:** {gene_identified}")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")
    st.write(f"**Identity (% of input):** {best_identity:.2f}")
    st.write(f"**Coverage (% of input aligned):** {best_coverage:.2f}")
    st.write(f"**ORF Check:** {orf_status}")
    st.write(f"**Critical Mutation:** {critical_mutation}")
    st.markdown(f"### QC Assessment: {qc_assessment}")

    # Extra clarity: what "query-centric" means numerically
    with st.expander("Show query-centric breakdown"):
        st.write(f"Input length: **{len(Q)} bp**")
        st.write(f"Aligned input bases: **{aligned_bases} / {len(Q)}**")
        st.write(f"Matched bases (within aligned region): **{matches} / {len(Q)}**")
        st.write(f"Uncovered (not aligned) bases: **{len(uncovered_pos)} / {len(Q)}**")

    # =====================================================
    # Highlight mismatch positions
    # =====================================================
    if show_positions:
        with st.expander("Highlight mismatch/uncovered positions (1-based on input)"):
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
                    # show first 200 positions to avoid huge UI
                    preview = uncovered_pos[:200]
                    tail_note = "" if len(uncovered_pos) <= 200 else f"\n... (+{len(uncovered_pos)-200} more)"
                    st.code(", ".join(map(str, preview)) + tail_note)
                else:
                    st.write("None")

        # Optional: show an annotated sequence view (simple)
        with st.expander("Annotated input sequence (mismatch=lowercase, uncovered=N)"):
            mismatch_set = set(mismatch_pos)
            uncovered_set = set(uncovered_pos)
            out_chars = []
            for i, base in enumerate(Q, start=1):
                if i in uncovered_set:
                    out_chars.append("N")
                elif i in mismatch_set:
                    out_chars.append(base.lower())
                else:
                    out_chars.append(base)
            # format in blocks of 60
            annotated = "\n".join("".join(out_chars[i:i+60]) for i in range(0, len(out_chars), 60))
            st.code(annotated, language="text")
            st.caption("Legend: UPPERCASE=match, lowercase=mismatch, N=uncovered (no alignment)")

    # =====================================================
    # Download QC report (CSV)
    # =====================================================
    report = pd.DataFrame([{
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene_identified,
        "Subtype": SUBTYPE_LABEL,
        "Identity (% of input)": round(best_identity, 2),
        "Coverage (% of input aligned)": round(best_coverage, 2),
        "Matched bases": matches,
        "Aligned input bases": aligned_bases,
        "Input length (bp)": len(Q),
        "Mismatch positions (1-based input)": ";".join(map(str, mismatch_pos)) if mismatch_pos else "",
        "Uncovered positions (1-based input)": ";".join(map(str, uncovered_pos)) if uncovered_pos else "",
        "ORF Check": orf_status,
        "Critical Mutation": critical_mutation,
        "QC Flag": qc_flag,
        "QC Assessment": qc_assessment,
        "Reference used": gene_label
    }])

    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download QC Report (CSV)",
        data=csv_bytes,
        file_name=f"{sample_id or 'qc_report'}_HA_NA_identity.csv",
        mime="text/csv"
    )

st.caption(
    "QC note: Identity/Coverage are QUERY-centric (the INPUT sequence is the denominator). "
    "Uncovered input bases are counted as non-matching in Identity by design."
)
