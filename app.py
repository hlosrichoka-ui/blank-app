import re
import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

# =========================
# Reference sequences (user-provided)
# =========================
HA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 4 hemagglutinin (HA) gene"
NA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 6 neuraminidase (NA) gene"
SUBTYPE_LABEL = "H7N9"
ORGANISM_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9))"

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

# =========================
# Helpers
# =========================
IUPAC_TO_N = re.compile(r"[^ACGTNacgtn]")

def clean_nt(seq: str) -> str:
    """
    Keep A/C/G/T; convert IUPAC ambiguity to N (do not delete, to avoid shifting).
    """
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines).replace(" ", "").replace("\t", "").upper()
    joined = IUPAC_TO_N.sub("N", joined)
    joined = re.sub(r"[^ACGTN]", "N", joined)
    return joined

def translate_frame(nt: str, frame: int) -> str:
    trimmed = nt[frame:]
    trimmed = trimmed[: (len(trimmed) // 3) * 3]
    if not trimmed:
        return ""
    return str(Seq(trimmed).translate(to_stop=False))

def best_orf_6frames(nt: str):
    candidates = []
    fwd = nt
    rev = str(Seq(nt).reverse_complement())
    for strand, seq in [("forward", fwd), ("reverse_complement", rev)]:
        for frame in [0, 1, 2]:
            aa = translate_frame(seq, frame)
            pre = aa.split("*")[0] if aa else ""
            candidates.append({
                "strand": strand,
                "frame": frame + 1,
                "aa_len_before_stop": len(pre),
                "stop_count": aa.count("*") if aa else 0,
            })
    return sorted(candidates, key=lambda x: (x["aa_len_before_stop"], -x["stop_count"]), reverse=True)[0]

def summarize_ranges(ranges):
    """
    ranges: list of (start,end)
    returns (min_start, max_end, total_len)
    """
    if not ranges:
        return None, None, 0
    min_s = min(s for s, e in ranges)
    max_e = max(e for s, e in ranges)
    total = sum((e - s) for s, e in ranges)
    return min_s, max_e, total

def local_align_best_region(ref: str, qry: str):
    """
    Local alignment to find best matching region (not necessarily from base 1).
    Returns a dict with identity, coverages, and coordinates on query/ref.
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    alns = aligner.align(ref, qry)
    if len(alns) == 0:
        return {
            "identity_pct": 0.0,
            "qry_cov_pct": 0.0,
            "ref_cov_pct": 0.0,
            "aligned_pairs": 0,
            "qry_start": None,
            "qry_end": None,
            "ref_start": None,
            "ref_end": None,
            "score": 0.0,
        }

    aln = alns[0]

    # Coordinates of aligned blocks for each sequence
    # aln.aligned = (ref_ranges, qry_ranges)
    ref_ranges = [tuple(x) for x in aln.aligned[0]]
    qry_ranges = [tuple(x) for x in aln.aligned[1]]

    ref_start, ref_end, ref_aligned_bases = summarize_ranges(ref_ranges)
    qry_start, qry_end, qry_aligned_bases = summarize_ranges(qry_ranges)

    # Compute identity using formatted alignment columns (ignore gaps)
    s = aln.format().splitlines()
    ref_aln = s[0].replace(" ", "")
    qry_aln = s[2].replace(" ", "")

    matches = 0
    aligned_pairs = 0
    for r, q in zip(ref_aln, qry_aln):
        if r != "-" and q != "-":
            aligned_pairs += 1
            if r == q:
                matches += 1

    identity = (matches / aligned_pairs * 100) if aligned_pairs else 0.0
    qry_cov = (qry_aligned_bases / len(qry) * 100) if len(qry) else 0.0
    ref_cov = (ref_aligned_bases / len(ref) * 100) if len(ref) else 0.0

    return {
        "identity_pct": identity,
        "qry_cov_pct": qry_cov,
        "ref_cov_pct": ref_cov,
        "aligned_pairs": aligned_pairs,
        "qry_start": qry_start,
        "qry_end": qry_end,
        "ref_start": ref_start,
        "ref_end": ref_end,
        "score": float(aln.score),
        "ref_ranges": ref_ranges,
        "qry_ranges": qry_ranges,
    }

# =========================
# App UI
# =========================
st.subheader("HA/NA Best-Match Region Finder (Local alignment)")

sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)
query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

IDENTITY_PASS = 95.0
QRY_COVERAGE_PASS = 90.0

if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q or set(Q) <= {"N"}:
        st.error("Please paste a valid nucleotide sequence (A/T/G/C; IUPAC allowed).")
        st.stop()

    ha_ref = clean_nt(HA_REF_RAW)
    na_ref = clean_nt(NA_REF_RAW)

    ha = local_align_best_region(ha_ref, Q)
    na = local_align_best_region(na_ref, Q)

    # choose by (identity, query coverage, aligned_pairs, score)
    if (ha["identity_pct"], ha["qry_cov_pct"], ha["aligned_pairs"], ha["score"]) >= (
        na["identity_pct"], na["qry_cov_pct"], na["aligned_pairs"], na["score"]
    ):
        gene_identified = "HA"
        best = ha
        gene_label = HA_REF_LABEL
    else:
        gene_identified = "NA"
        best = na
        gene_label = NA_REF_LABEL

    best_orf = best_orf_6frames(Q)
    orf_investigate = (best_orf["aa_len_before_stop"] < 80) or (best_orf["stop_count"] > 1)
    orf_status = "PASS" if not orf_investigate else "INVESTIGATE"
    critical_mutation = "None (not assessed in Phase 1)"

    # QC decision
    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
        qc_ok = False
    elif best["identity_pct"] < IDENTITY_PASS or best["qry_cov_pct"] < QRY_COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low identity / query coverage)"
        qc_ok = False
    elif orf_investigate:
        qc_assessment = "⚠️ INVESTIGATE (ORF check)"
        qc_ok = False
    else:
        qc_assessment = "✅ PASS"
        qc_ok = True

    # ---- Display ----
    st.markdown("---")
    st.subheader("🔍 Analysis Result (Best matching region)")

    st.write(f"**Organism:** {ORGANISM_LABEL}")
    st.write(f"**Gene Identified:** {gene_identified} — ({gene_label})")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")

    st.write(f"**Identity (% within matched region):** {best['identity_pct']:.2f}")
    st.write(f"**Query coverage (%):** {best['qry_cov_pct']:.2f}")
    st.write(f"**Aligned length (bp, no-gap pairs):** {best['aligned_pairs']}")

    # Coordinates (0-based internal -> show 1-based to users)
    if best["qry_start"] is not None:
        st.write(
            f"**Best-hit coordinates (1-based):** "
            f"Query {best['qry_start']+1}–{best['qry_end']}  |  "
            f"Ref {best['ref_start']+1}–{best['ref_end']}"
        )
        st.caption(f"Blocks (ref): {best['ref_ranges']}  |  Blocks (query): {best['qry_ranges']}")

    st.write(f"**ORF Check:** {orf_status}")
    st.write(f"**Critical Mutation:** {critical_mutation}")
    st.markdown(f"### QC Assessment: {qc_assessment}")

    # Optional: show alignment text
    with st.expander("Show alignment (best local hit)"):
        # Recompute to get the actual alignment object text easily
        # (quick way: run aligner again for the chosen gene)
        aligner = PairwiseAligner()
        aligner.mode = "local"
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -3
        aligner.extend_gap_score = -0.5
        ref = ha_ref if gene_identified == "HA" else na_ref
        aln = aligner.align(ref, Q)[0]
        st.text(aln.format())

    # ---- CSV ----
    report = pd.DataFrame([{
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene_identified,
        "Subtype": SUBTYPE_LABEL,
        "Identity (%)": round(best["identity_pct"], 2),
        "Query coverage (%)": round(best["qry_cov_pct"], 2),
        "Aligned length (bp)": int(best["aligned_pairs"]),
        "Query start (1-based)": None if best["qry_start"] is None else int(best["qry_start"] + 1),
        "Query end (1-based)": None if best["qry_end"] is None else int(best["qry_end"]),
        "Ref start (1-based)": None if best["ref_start"] is None else int(best["ref_start"] + 1),
        "Ref end (1-based)": None if best["ref_end"] is None else int(best["ref_end"]),
        "ORF Check": orf_status,
        "Critical Mutation": critical_mutation,
        "QC Assessment": qc_assessment,
        "Reference (HA)": HA_REF_LABEL,
        "Reference (NA)": NA_REF_LABEL
    }])

    st.download_button(
        label="Download QC Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name=f"{sample_id or 'qc_report'}_HA_NA_best_hit.csv",
        mime="text/csv"
    )

    with st.expander("Show QC thresholds used"):
        st.write(f"Identity PASS threshold: {IDENTITY_PASS}%")
        st.write(f"Query coverage PASS threshold: {QRY_COVERAGE_PASS}%")
        st.write("Identity is computed only within the matched region (columns where both bases are not gaps).")

st.caption("QC note: This tool finds the best local matching region against the provided HA/NA references (BLAST-like screening).")
