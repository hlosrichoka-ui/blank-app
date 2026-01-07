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
    Keep A/C/G/T and convert IUPAC ambiguity (e.g., Y, R, K, M, S, W, B, D, H, V) to N.
    Do NOT delete them (deleting shifts positions and can distort alignment).
    """
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines).replace(" ", "").replace("\t", "")
    joined = joined.upper()
    joined = IUPAC_TO_N.sub("N", joined)
    # also remove anything still weird like digits/punct that may remain (turn into N then trim)
    joined = re.sub(r"[^ACGTN]", "N", joined)
    return joined

def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return 100.0 * (seq.count("G") + seq.count("C")) / len(seq)

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
                "aa": aa,
                "aa_len_before_stop": len(pre),
                "stop_count": aa.count("*") if aa else 0,
            })
    best = sorted(
        candidates,
        key=lambda x: (x["aa_len_before_stop"], -x["stop_count"]),
        reverse=True
    )[0]
    return best

def align_and_score_local(ref: str, qry: str):
    """
    Local alignment to mimic BLAST-like "best matching region".
    Returns:
      identity_pct over aligned columns where neither is gap
      query_coverage_pct = aligned_query_bases / len(query)
      ref_coverage_pct   = aligned_ref_bases / len(ref)
      aligned_len (columns where neither is gap)
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    alns = aligner.align(ref, qry)
    if len(alns) == 0:
        return 0.0, 0.0, 0.0, 0

    aln = alns[0]
    s = aln.format().splitlines()
    # Biopython format usually:
    # line0: ref with gaps
    # line1: match markers
    # line2: qry with gaps
    ref_aln = s[0].replace(" ", "")
    qry_aln = s[2].replace(" ", "")

    matches = 0
    aligned_pairs = 0
    aligned_q_bases = 0
    aligned_r_bases = 0

    for r, q in zip(ref_aln, qry_aln):
        r_gap = (r == "-")
        q_gap = (q == "-")
        if not q_gap:
            aligned_q_bases += 1
        if not r_gap:
            aligned_r_bases += 1
        if (not r_gap) and (not q_gap):
            aligned_pairs += 1
            if r == q:
                matches += 1

    identity = (matches / aligned_pairs * 100) if aligned_pairs else 0.0
    q_cov = (aligned_q_bases / len(qry) * 100) if len(qry) else 0.0
    r_cov = (aligned_r_bases / len(ref) * 100) if len(ref) else 0.0
    return identity, q_cov, r_cov, aligned_pairs

# =========================
# App UI
# =========================
st.subheader("HA/NA Sequence Identity Quick-Check (Local alignment)")

sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)
query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

# Thresholds (QC rule) – can adjust
IDENTITY_PASS = 95.0
QRY_COVERAGE_PASS = 90.0   # how much of the query is covered by best local hit
REF_COVERAGE_PASS = 40.0   # optional: if query is amplicon/partial, ref coverage may be low; keep modest

if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q or set(Q) <= {"N"}:
        st.error("Please paste a valid nucleotide sequence (A/T/G/C; IUPAC allowed).")
        st.stop()

    ha_ref = clean_nt(HA_REF_RAW)
    na_ref = clean_nt(NA_REF_RAW)

    # Compare to refs (LOCAL)
    ha_id, ha_qcov, ha_rcov, ha_alnlen = align_and_score_local(ha_ref, Q)
    na_id, na_qcov, na_rcov, na_alnlen = align_and_score_local(na_ref, Q)

    # Decide gene by (identity, query coverage, aligned length)
    if (ha_id, ha_qcov, ha_alnlen) >= (na_id, na_qcov, na_alnlen):
        gene_identified = "HA"
        best_identity = ha_id
        best_qcov = ha_qcov
        best_rcov = ha_rcov
        best_alnlen = ha_alnlen
        gene_label = HA_REF_LABEL
    else:
        gene_identified = "NA"
        best_identity = na_id
        best_qcov = na_qcov
        best_rcov = na_rcov
        best_alnlen = na_alnlen
        gene_label = NA_REF_LABEL

    # ORF check (heuristic)
    best_orf = best_orf_6frames(Q)
    orf_investigate = (best_orf["aa_len_before_stop"] < 80) or (best_orf["stop_count"] > 1)
    orf_status = "PASS" if not orf_investigate else "INVESTIGATE"

    # Critical mutation (Phase 1 placeholder)
    critical_mutation = "None (not assessed in Phase 1)"

    # QC assessment rule
    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
        qc_ok = False
    elif best_identity < IDENTITY_PASS or best_qcov < QRY_COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low identity/query coverage)"
        qc_ok = False
    elif best_rcov < REF_COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low reference coverage — likely partial/short amplicon)"
        qc_ok = False
    elif orf_investigate:
        qc_assessment = "⚠️ INVESTIGATE (ORF check)"
        qc_ok = False
    else:
        qc_assessment = "✅ PASS"
        qc_ok = True

    # ---- Display results ----
    st.markdown("---")
    st.subheader("🔍 Analysis Result")
    st.write(f"**Organism:** {ORGANISM_LABEL}")
    st.write(f"**Gene Identified:** {gene_identified}  —  ({gene_label})")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")

    st.write(f"**Identity (%):** {best_identity:.2f}")
    st.write(f"**Query Coverage (%):** {best_qcov:.2f}")
    st.write(f"**Reference Coverage (%):** {best_rcov:.2f}")
    st.write(f"**Aligned length (bp, non-gap pairs):** {best_alnlen}")

    st.write(f"**ORF Check:** {orf_status}")
    st.write(f"**Critical Mutation:** {critical_mutation}")
    st.markdown(f"### QC Assessment: {qc_assessment}")

    # ---- Download report (CSV) ----
    report = pd.DataFrame([{
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene_identified,
        "Subtype": SUBTYPE_LABEL,
        "Identity (%)": round(best_identity, 2),
        "Query Coverage (%)": round(best_qcov, 2),
        "Reference Coverage (%)": round(best_rcov, 2),
        "Aligned length (bp)": best_alnlen,
        "ORF Check": orf_status,
        "Critical Mutation": critical_mutation,
        "QC Assessment": qc_assessment,
        "Reference (HA)": HA_REF_LABEL,
        "Reference (NA)": NA_REF_LABEL
    }])

    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download QC Report (CSV)",
        data=csv_bytes,
        file_name=f"{sample_id or 'qc_report'}_HA_NA_identity.csv",
        mime="text/csv"
    )

    with st.expander("Show QC thresholds used"):
        st.write(f"Identity PASS threshold: {IDENTITY_PASS}%")
        st.write(f"Query coverage PASS threshold: {QRY_COVERAGE_PASS}%")
        st.write(f"Reference coverage PASS threshold: {REF_COVERAGE_PASS}%")
        st.write("ORF investigate rule: best frame AA before first stop < 80 OR stop_count > 1")

st.caption("QC note: Local alignment is used to approximate a BLAST-like best-hit screening against the provided HA/NA references.")
