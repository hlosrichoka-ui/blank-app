import re
import io
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
def clean_nt(seq: str) -> str:
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines)
    # keep only ATGC (drop IUPAC ambiguity like Y)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

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

def align_and_score(ref: str, qry: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(ref, qry)))
    lines = aln.format().splitlines()
    ref_aln = lines[0].replace(" ", "")
    qry_aln = lines[2].replace(" ", "")

    # identity over query aligned bases (non-gap in query)
    matches = 0
    aligned_q = 0
    for r, q in zip(ref_aln, qry_aln):
        if q != "-":
            aligned_q += 1
            if r == q:
                matches += 1

    identity = (matches / aligned_q * 100) if aligned_q else 0.0
    coverage = (aligned_q / len(qry) * 100) if len(qry) else 0.0
    return identity, coverage

# =========================
# App UI (matching your layout)
# =========================
st.subheader("HA/NA Sequence Identity Quick-Check")

sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)

query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

# Thresholds (QC rule) – can adjust
IDENTITY_PASS = 95.0
COVERAGE_PASS = 90.0

if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q:
        st.error("Please paste a valid nucleotide sequence (A/T/G/C).")
        st.stop()

    ha_ref = clean_nt(HA_REF_RAW)
    na_ref = clean_nt(NA_REF_RAW)

    # Compare to refs
    ha_id, ha_cov = align_and_score(ha_ref, Q)
    na_id, na_cov = align_and_score(na_ref, Q)

    # Decide gene by better (identity, coverage)
    if (ha_id, ha_cov) >= (na_id, na_cov):
        gene_identified = "HA"
        best_identity = ha_id
        best_coverage = ha_cov
        gene_label = HA_REF_LABEL
    else:
        gene_identified = "NA"
        best_identity = na_id
        best_coverage = na_cov
        gene_label = NA_REF_LABEL

    # ORF check (heuristic)
    best_orf = best_orf_6frames(Q)
    orf_investigate = (best_orf["aa_len_before_stop"] < 80) or (best_orf["stop_count"] > 1)
    orf_status = "PASS" if not orf_investigate else "INVESTIGATE"

    # Critical mutation (Phase 1 placeholder)
    critical_mutation = "None (not assessed in Phase 1)"

    # QC assessment rule
    # Fail if gene mismatch vs expected OR if identity/coverage below thresholds
    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
        qc_ok = False
    elif best_identity < IDENTITY_PASS or best_coverage < COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low identity/coverage)"
        qc_ok = False
    elif orf_investigate:
        qc_assessment = "⚠️ INVESTIGATE (ORF check)"
        qc_ok = False
    else:
        qc_assessment = "✅ PASS"
        qc_ok = True

    # ---- Display results (as requested) ----
    st.markdown("---")
    st.subheader("🔍 Analysis Result")
    st.write(f"**Organism:** {ORGANISM_LABEL}")
    st.write(f"**Gene Identified:** {gene_identified}  —  ({gene_label})")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")
    st.write(f"**Identity (%):** {best_identity:.2f}")
    st.write(f"**Coverage (%):** {best_coverage:.2f}")
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
        "Coverage (%)": round(best_coverage, 2),
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

    # Optional: show thresholds used
    with st.expander("Show QC thresholds used"):
        st.write(f"Identity PASS threshold: {IDENTITY_PASS}%")
        st.write(f"Coverage PASS threshold: {COVERAGE_PASS}%")
        st.write("ORF investigate rule: best frame AA before first stop < 80 OR stop_count > 1")

st.caption("QC note: This tool compares only to the provided HA/NA references and is intended for screening/supporting evidence.")
