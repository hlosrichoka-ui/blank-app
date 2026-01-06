import re
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
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.startswith(">")]
    joined = "".join(lines)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

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

def query_centric_local_compare(ref: str, qry: str):
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(ref, qry)))
    q_blocks = aln.aligned[1]
    total_query = len(qry)

    if q_blocks is None or len(q_blocks) == 0:
        return 0.0, 0.0, 0.0, [], list(range(1, total_query + 1)), 0, 0

    covered = set()
    for qs, qe in q_blocks:
        for p in range(qs + 1, qe + 1):
            covered.add(p)

    uncovered = [p for p in range(1, total_query + 1) if p not in covered]

    lines = aln.format().splitlines()
    ref_aln = lines[0].replace(" ", "")
    qry_aln = lines[2].replace(" ", "")

    covered_sorted = sorted(covered)
    idx = 0
    matches = 0
    aligned = 0
    mismatches = []

    for r, q in zip(ref_aln, qry_aln):
        if q == "-":
            continue
        if idx >= len(covered_sorted):
            break
        pos = covered_sorted[idx]
        idx += 1
        aligned += 1
        if r == q:
            matches += 1
        else:
            mismatches.append(pos)

    identity_input = matches / total_query * 100
    identity_aligned = matches / aligned * 100 if aligned else 0.0
    coverage = aligned / total_query * 100

    return identity_input, identity_aligned, coverage, mismatches, uncovered, matches, aligned

# =========================================================
# UI
# =========================================================
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)
query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220)

IDENTITY_PASS = 95.0
COVERAGE_PASS = 90.0

if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q:
        st.error("Invalid sequence")
        st.stop()

    HA_REF = clean_nt(HA_REF_RAW)
    NA_REF = clean_nt(NA_REF_RAW)

    ha = query_centric_local_compare(HA_REF, Q)
    na = query_centric_local_compare(NA_REF, Q)

    if (ha[0], ha[2]) >= (na[0], na[2]):
        gene = "HA"
        res = ha
        ref_label = HA_REF_LABEL
    else:
        gene = "NA"
        res = na
        ref_label = NA_REF_LABEL

    identity_input, identity_aligned, coverage, mismatches, uncovered, matches, aligned = res
    best_len, stop_count = best_orf_6frames(Q)
    orf_status = "PASS" if best_len >= 80 and stop_count <= 1 else "INVESTIGATE"

    if gene != expected_gene:
        qc = "❌ FAIL (Gene mismatch)"
    elif identity_input < IDENTITY_PASS or coverage < COVERAGE_PASS:
        qc = "⚠️ INVESTIGATE (Low identity/coverage)"
    elif orf_status != "PASS":
        qc = "⚠️ INVESTIGATE (ORF check)"
    else:
        qc = "✅ PASS"

    st.markdown("---")
    st.subheader("🔍 Analysis Result")
    st.write(f"Gene Identified: **{gene}**")
    st.write(f"Subtype: **{SUBTYPE_LABEL}**")
    st.write(f"Identity (% of input): **{identity_input:.2f}**")
    st.write(f"Identity (% of aligned region): **{identity_aligned:.2f}**")
    st.write(f"Coverage (% of input aligned): **{coverage:.2f}**")
    st.write(f"ORF Check: **{orf_status}**")
    st.markdown(f"### QC Assessment: {qc}")

    with st.expander("Mismatch / Uncovered positions"):
        st.write("Mismatch positions:", mismatches if mismatches else "None")
        st.write("Uncovered positions:", uncovered if uncovered else "None")

    report = pd.DataFrame([{
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene,
        "Identity (% input)": round(identity_input, 2),
        "Identity (% aligned)": round(identity_aligned, 2),
        "Coverage (% input)": round(coverage, 2),
        "QC Assessment": qc,
        "Reference": ref_label
    }])

    st.download_button(
        "Download QC Report (CSV)",
        report.to_csv(index=False).encode(),
        file_name=f"{sample_id or 'qc'}_HA_NA_identity.csv",
        mime="text/csv"
    )

st.caption(
    "Identity (% of input) = sample ตรงกับ reference กี่ % ของ input ทั้งหมด "
    "(query-centric QC logic)"
)
# alignment coordinates in original sequences (0-based half-open)
ref_blocks = aln.aligned[0]
qry_blocks = aln.aligned[1]

# convert to 1-based inclusive coordinates (nice for QC report)
ref_ranges = [(rs+1, re) for rs, re in ref_blocks]
qry_ranges = [(qs+1, qe) for qs, qe in qry_blocks]

