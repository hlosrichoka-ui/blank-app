import re
import io
import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="HA/NA Sequence Identity Quick-Check",
    layout="centered"
)

st.title("HA/NA Sequence Identity Quick-Check")

# =========================================================
# Reference information
# =========================================================
ORGANISM_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9))"
SUBTYPE_LABEL = "H7N9"

HA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 4 hemagglutinin (HA) gene"
NA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 6 neuraminidase (NA) gene"

# -------------------------
# Reference sequences
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
# Helper functions
# =========================================================
def clean_nt(seq: str) -> str:
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.startswith(">")]
    joined = "".join(lines)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

def best_orf_6frames(nt: str):
    best_len = 0
    stop_count = 0
    for frame in [0, 1, 2]:
        aa = str(Seq(nt[frame:]).translate(to_stop=False))
        pre = aa.split("*")[0]
        if len(pre) > best_len:
            best_len = len(pre)
            stop_count = aa.count("*")
    return best_len, stop_count

def local_align_query_centric(ref: str, qry: str):
    """
    Local alignment.
    Identity and coverage are calculated based on FULL QUERY length.
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(ref, qry)))
    lines = aln.format().splitlines()
    ref_aln = lines[0].replace(" ", "")
    qry_aln = lines[2].replace(" ", "")

    matches = 0
    aligned_q = 0
    mismatch_positions = []

    q_pos = 0
    for r, q in zip(ref_aln, qry_aln):
        if q != "-":
            q_pos += 1
            aligned_q += 1
            if r == q:
                matches += 1
            else:
                mismatch_positions.append(q_pos)

    identity = (matches / len(qry)) * 100 if len(qry) else 0
    coverage = (aligned_q / len(qry)) * 100 if len(qry) else 0

    return identity, coverage, mismatch_positions

# =========================================================
# UI input
# =========================================================
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)

query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

IDENTITY_PASS = 95.0
COVERAGE_PASS = 90.0

# =========================================================
# Analysis
# =========================================================
if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q:
        st.error("Invalid nucleotide sequence.")
        st.stop()

    HA_REF = clean_nt(HA_REF_RAW)
    NA_REF = clean_nt(NA_REF_RAW)

    ha_id, ha_cov, ha_mismatch = local_align_query_centric(HA_REF, Q)
    na_id, na_cov, na_mismatch = local_align_query_centric(NA_REF, Q)

    if (ha_id, ha_cov) >= (na_id, na_cov):
        gene_identified = "HA"
        best_identity = ha_id
        best_coverage = ha_cov
        mismatch_pos = ha_mismatch
        gene_label = HA_REF_LABEL
    else:
        gene_identified = "NA"
        best_identity = na_id
        best_coverage = na_cov
        mismatch_pos = na_mismatch
        gene_label = NA_REF_LABEL

    best_len, stop_count = best_orf_6frames(Q)
    orf_status = "PASS" if (best_len >= 80 and stop_count <= 1) else "INVESTIGATE"

    if gene_identified != expected_gene:
        qc_assessment = "❌ FAIL (Gene mismatch)"
    elif best_identity < IDENTITY_PASS or best_coverage < COVERAGE_PASS:
        qc_assessment = "⚠️ INVESTIGATE (Low identity/coverage)"
    elif orf_status != "PASS":
        qc_assessment = "⚠️ INVESTIGATE (ORF check)"
    else:
        qc_assessment = "✅ PASS"

    # =====================================================
    # Display result (layout requested)
    # =====================================================
    st.markdown("---")
    st.subheader("🔍 Analysis Result")
    st.write(f"**Gene Identified:** {gene_identified}")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")
    st.write(f"**Identity (% of input):** {best_identity:.2f}")
    st.write(f"**Coverage (% of input aligned):** {best_coverage:.2f}")
    st.write(f"**ORF Check:** {orf_status}")
    st.write(f"**Critical Mutation:** {'None' if not mismatch_pos else 'See mismatch positions'}")

    st.markdown(f"### QC Assessment: {qc_assessment}")

    with st.expander("Show mismatch positions (query-based)"):
        if mismatch_pos:
            st.write(f"Total mismatches: {len(mismatch_pos)}")
            st.code(", ".join(map(str, mismatch_pos)))
        else:
            st.write("No mismatches detected.")

    # =====================================================
    # Download QC report
    # =====================================================
    report = pd.DataFrame([{
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene_identified,
        "Subtype": SUBTYPE_LABEL,
        "Identity (% of input)": round(best_identity, 2),
        "Coverage (% of input)": round(best_coverage, 2),
        "ORF Check": orf_status,
        "Mismatch positions (query)": ";".join(map(str, mismatch_pos)) if mismatch_pos else "",
        "QC Assessment": qc_assessment,
        "Reference used": gene_label
    }])

    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download QC Report (CSV)",
        data=csv_bytes,
        file_name=f"{sample_id or 'qc'}_HA_NA_identity.csv",
        mime="text/csv"
    )

st.caption(
    "QC note: Identity and coverage are calculated in a query-centric manner. "
    "This tool is intended for screening/supporting evidence only."
)
