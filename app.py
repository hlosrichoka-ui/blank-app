import re
import json
import base64
import urllib.request
import urllib.error

import pandas as pd
import streamlit as st

from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# =========================================================
# Page config (ONLY ONCE)
# =========================================================
st.set_page_config(page_title="HA/NA QC + DNA→Protein + 3D", layout="centered")
st.title("🧬 QC Sequence Toolbox")

# =========================================================
# Sidebar: App selector
# =========================================================
app_mode = st.sidebar.selectbox(
    "Select module",
    [
        "HA/NA Sequence Identification Quick-Check",
        "DNA → RNA → Protein + Free 3D Viewer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Streamlit Cloud runs 1 entrypoint (app.py). Use this selector to switch modules.")

# =========================================================
# ===================== MODULE 1 ==========================
# HA/NA Sequence Identification Quick-Check
# =========================================================

ORGANISM_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9))"
SUBTYPE_LABEL = "H7N9"

HA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 4 hemagglutinin (HA) gene"
NA_REF_LABEL = "Influenza A virus (A/Shanghai/02/2013(H7N9)) segment 6 neuraminidase (NA) gene"

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

def best_orf_6frames_nt(nt: str):
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

    %Identification (matches/input_length)*100  <-- PASS when >= 95
    Identity (aligned region)                  <-- BLAST-like
    Coverage (of input aligned)
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

    # 1-based inclusive display ranges
    ref_ranges = [(int(rs) + 1, int(re)) for rs, re in ref_blocks]
    qry_ranges = [(int(qs) + 1, int(qe)) for qs, qe in qry_blocks]

    matches = 0
    mismatches = []
    insertions = []
    covered = set()
    aligned_query_bases = 0

    prev_rs = prev_re = prev_qs = prev_qe = None

    for (rs, re), (qs, qe) in zip(ref_blocks, qry_blocks):
        rs, re, qs, qe = int(rs), int(re), int(qs), int(qe)

        # gaps between blocks => insertion or deletion
        if prev_rs is not None:
            r_gap = rs - prev_re
            q_gap = qs - prev_qe

            # insertion in query: q advances but ref doesn't
            if q_gap > 0 and r_gap == 0:
                for p0 in range(prev_qe, qs):
                    qpos_1based = p0 + 1
                    insertions.append(qpos_1based)
                    covered.add(qpos_1based)
                aligned_query_bases += q_gap

        # aligned block comparison
        block_len = min(re - rs, qe - qs)
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

def module_ha_na():
    st.header("HA/NA Sequence Identification Quick-Check")

    sample_id = st.text_input("Sample ID", key="ha_sample")
    expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True, key="ha_expected")
    query_raw = st.text_area("Paste Nucleotide Sequence (FASTA):", height=220, key="ha_query")

    IDENTIFICATION_PASS = st.number_input("%Identification PASS threshold (matches/input length)", value=95.0, step=0.5, key="ha_ident_thr")
    COVERAGE_PASS = st.number_input("Coverage PASS threshold (% of input aligned)", value=90.0, step=1.0, key="ha_cov_thr")

    show_positions = st.checkbox("Show mismatch/insertion/uncovered positions", value=True, key="ha_pos")
    show_annotated = st.checkbox("Show annotated input sequence", value=True, key="ha_anno")
    show_ranges = st.checkbox("Show aligned start/end positions (REF & INPUT)", value=True, key="ha_ranges")

    if st.button("Analyze Sequence", type="primary", key="ha_btn"):
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
        best_len, stop_count = best_orf_6frames_nt(Q)
        orf_status = "PASS" if (best_len >= 80 and stop_count <= 1) else "INVESTIGATE"

        # QC Assessment
        ident_check = round(ident_pct, 2)
        cov_check = round(cov_pct, 2)

        if gene_identified != expected_gene:
            qc_assessment = "❌ FAIL (Gene mismatch)"
            qc_flag = "FAIL"
        elif ident_check >= IDENTIFICATION_PASS and cov_check >= COVERAGE_PASS and orf_status == "PASS":
            qc_assessment = "✅ PASS"
            qc_flag = "PASS"
        else:
            reasons = []
            if ident_check < IDENTIFICATION_PASS:
                reasons.append("Low %Identification")
            if cov_check < COVERAGE_PASS:
                reasons.append("Low Coverage")
            if orf_status != "PASS":
                reasons.append("ORF check")
            qc_assessment = "⚠️ INVESTIGATE (" + ", ".join(reasons) + ")"
            qc_flag = "INVESTIGATE"

        # Display results
        st.markdown("---")
        st.subheader("🔍 Analysis Result")
        st.write(f"**Sample ID:** {sample_id if sample_id else '-'}")
        st.write(f"**Organism:** {ORGANISM_LABEL}")
        st.write(f"**Gene Identified:** {gene_identified} — ({gene_label})")
        st.write(f"**Subtype:** {SUBTYPE_LABEL}")

        st.write(f"**%Identification (matches / input length):** {ident_pct:.2f}")
        st.caption("= (จำนวนเบสที่เหมือนกัน / จำนวนเบสของ input ทั้งหมด) × 100  | PASS when ≥ threshold")

        st.write(f"**Identity (% of aligned region):** {ident_aligned:.2f}")
        st.caption("= ตรงกันกี่ % เฉพาะส่วนที่ align ได้ (BLAST-like)")

        st.write(f"**Coverage (% of input aligned):** {cov_pct:.2f}")
        st.caption(f"Decision values (rounded): %Identification={ident_check:.2f}, Coverage={cov_check:.2f}")

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

        # Download QC report (CSV)
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
            mime="text/csv",
            key="ha_csv",
        )

    st.caption("LOCAL alignment; %Identification = matches / input length × 100 (PASS when ≥ threshold; e.g., 95% = PASS).")


# =========================================================
# ===================== MODULE 2 ==========================
# DNA → RNA → Protein + Free 3D Viewer
# =========================================================

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
ALPHAFOLD_ENTRY = "https://www.alphafold.ebi.ac.uk/entry"
COLABFOLD_WEB = "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"
AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

def clean_dna_keep_len(seq: str) -> str:
    """Remove FASTA headers; keep letters; convert non-ATGC to N (keep length stable)."""
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    s = "".join(lines).upper()
    s = re.sub(r"[^A-Z]", "", s)
    s = re.sub(r"[^ATGC]", "N", s)
    return s

def dna_to_rna(dna: str) -> str:
    """DNA -> RNA (T -> U), keep N as N."""
    return dna.replace("T", "U")

def translate_from_dna(dna: str, strand: str, frame: int) -> str:
    """
    Translate from DNA using Biopython.
    - strand '+' uses dna as-is
    - strand '-' uses reverse complement of dna
    - frame: 0/1/2
    returns AA string including '*'
    """
    seq = dna
    if strand == "-":
        seq = str(Seq(dna).reverse_complement())
    seq = seq[frame:]
    seq = seq[: (len(seq)//3)*3]
    return str(Seq(seq).translate(to_stop=False)) if seq else ""

def best_orf_6frames_dna(dna: str):
    """Pick longest AA before first stop among 6 frames."""
    best = {"strand": "+", "frame": 0, "aa_full": "", "aa_orf": "", "orf_len": 0, "stop_count": 10**9}
    for strand in ["+", "-"]:
        for frame in [0, 1, 2]:
            aa_full = translate_from_dna(dna, strand, frame)
            if not aa_full:
                continue
            aa_orf = aa_full.split("*")[0]
            orf_len = len(aa_orf)
            stop_count = aa_full.count("*")
            if (orf_len > best["orf_len"]) or (orf_len == best["orf_len"] and stop_count < best["stop_count"]):
                best = {
                    "strand": strand,
                    "frame": frame,
                    "aa_full": aa_full,
                    "aa_orf": aa_orf,
                    "orf_len": orf_len,
                    "stop_count": stop_count,
                }
    return best

def sanitize_protein(aa: str) -> str:
    aa = aa.upper()
    return "".join([c for c in aa if c in AA_VALID])

def kyte_doolittle(aa: str, window=19):
    kd = {
        "I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,
        "G":-0.4,"T":-0.7,"S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,
        "H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5
    }
    aa = sanitize_protein(aa)
    if len(aa) < window:
        return [], []
    vals, pos = [], []
    for i in range(len(aa) - window + 1):
        seg = aa[i:i+window]
        vals.append(sum(kd[x] for x in seg) / window)
        pos.append(i + 1)  # 1-based window start
    return pos, vals

def http_get_json(url: str, timeout=60):
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))

def http_get_bytes(url: str, timeout=120):
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read()

def alphafold_lookup(uniprot_id: str):
    """
    AlphaFold DB prediction API:
    GET https://alphafold.ebi.ac.uk/api/prediction/<UniProt>
    Returns list with model file URLs (PDB/mmCIF) when available.
    """
    url = f"{ALPHAFOLD_API}/{uniprot_id}"
    return http_get_json(url)

def module_dna_protein_3d():
    st.header("DNA → RNA → Protein + Free 3D Viewer")

    sample_id = st.text_input("Sample ID", key="p_sample")
    dna_raw = st.text_area("Paste DNA sequence (FASTA / plain)", height=220, key="p_dna")

    mode = st.radio("Translation mode", ["Auto (best ORF)", "Manual"], horizontal=True, key="p_mode")
    c1, c2 = st.columns(2)
    strand = c1.selectbox("Strand", ["+", "-"], disabled=(mode == "Auto (best ORF)"), key="p_strand")
    frame = c2.selectbox("Frame", [0, 1, 2], disabled=(mode == "Auto (best ORF)"), key="p_frame")

    window = st.slider("Hydropathy window", 7, 31, 19, step=2, key="p_window")

    st.markdown("---")
    st.subheader("Free structure options")

    with st.expander("Option A (FREE): ColabFold / AlphaFold2 (run outside)"):
        st.write("ใช้ ColabFold (ฟรี) เพื่อทำนายโครงสร้างจาก protein sequence แล้วดาวน์โหลดไฟล์ PDB/mmCIF กลับมาอัปโหลดที่ Option C")
        st.link_button("Open ColabFold (AlphaFold2 notebook)", COLABFOLD_WEB)

    with st.expander("Option B (FREE if exists): AlphaFold DB by UniProt ID"):
        uniprot_id = st.text_input("UniProt Accession (e.g., Q8AVJ1)", value="", key="p_uniprot")
        st.caption("ถ้ามีโครงสร้างใน AlphaFold DB จะดึงไฟล์ PDB/mmCIF มาแสดง 3D ได้ทันที")

    with st.expander("Option C: Upload a structure file (PDB/mmCIF) to view in 3D"):
        uploaded = st.file_uploader("Upload .pdb or .cif", type=["pdb", "cif"], key="p_upload")

    if st.button("Convert & Analyze", type="primary", key="p_btn"):
        DNA = clean_dna_keep_len(dna_raw)
        if not DNA:
            st.error("Invalid DNA sequence.")
            st.stop()

        RNA = dna_to_rna(DNA)

        # Translate
        if mode == "Auto (best ORF)":
            best = best_orf_6frames_dna(DNA)
            strand_use = best["strand"]
            frame_use = best["frame"]
            aa_orf = best["aa_orf"]
            stop_count = best["stop_count"]
            note = f"Auto ORF: strand {strand_use}, frame {frame_use} | ORF={best['orf_len']} aa | stops(full)={stop_count}"
        else:
            strand_use = strand
            frame_use = int(frame)
            aa_full = translate_from_dna(DNA, strand_use, frame_use)
            aa_orf = aa_full.split("*")[0] if aa_full else ""
            stop_count = aa_full.count("*") if aa_full else 0
            note = f"Manual: strand {strand_use}, frame {frame_use} | ORF={len(aa_orf)} aa | stops(full)={stop_count}"

        protein = sanitize_protein(aa_orf)
        if not protein:
            st.error("Protein translation failed (check frame/strand or too many ambiguities).")
            st.stop()

        # Properties
        pa = ProteinAnalysis(protein)
        st.markdown("## ✅ Outputs")
        st.write(f"**Sample:** {sample_id or '-'}")
        st.write(f"**DNA length:** {len(DNA)} nt")
        st.write(f"**RNA length:** {len(RNA)} nt")
        st.write(f"**Translation:** {note}")
        st.write(f"**Protein length:** {len(protein)} aa")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MW (Da)", f"{pa.molecular_weight():,.1f}")
        m2.metric("pI", f"{pa.isoelectric_point():.2f}")
        m3.metric("GRAVY", f"{pa.gravy():.2f}")
        m4.metric("Aromaticity", f"{pa.aromaticity():.3f}")

        st.markdown("### DNA (cleaned)")
        st.code(DNA, language="text")

        st.markdown("### RNA (T→U)")
        st.code(RNA, language="text")

        st.markdown("### Protein (ORF before first stop)")
        fasta = f">{sample_id or 'protein'}|strand={strand_use}|frame={frame_use}\n{protein}\n"
        st.code(fasta, language="text")

        # Hydropathy plot (streamlit native)
        st.markdown("## 📈 Protein visualization (in-app)")
        pos, vals = kyte_doolittle(protein, window=window)
        if vals:
            df = pd.DataFrame({"Hydropathy (Kyte-Doolittle)": vals}, index=pos)
            st.line_chart(df)
        else:
            st.info("Protein too short for this hydropathy window.")

        # Downloads (DNA/RNA/Protein)
        st.download_button("Download DNA (txt)", DNA.encode("utf-8"), file_name=f"{sample_id or 'seq'}_DNA.txt", key="p_dna_dl")
        st.download_button("Download RNA (txt)", RNA.encode("utf-8"), file_name=f"{sample_id or 'seq'}_RNA.txt", key="p_rna_dl")
        st.download_button("Download Protein FASTA", fasta.encode("utf-8"), file_name=f"{sample_id or 'protein'}_protein.fasta", key="p_fasta_dl")

        # =====================================================
        # Structure Viewer (NGL)
        # =====================================================
        st.markdown("---")
        st.markdown("## 🧊 3D Structure Viewer (free routes)")

        structure_bytes = None
        structure_ext = None
        structure_source = None

        # Prefer uploaded file (Option C)
        if uploaded is not None:
            structure_bytes = uploaded.getvalue()
            structure_ext = uploaded.name.split(".")[-1].lower()
            structure_source = f"Uploaded file: {uploaded.name}"

        # Else try AlphaFold DB by UniProt (Option B)
        elif uniprot_id.strip():
            try:
                with st.spinner("Querying AlphaFold DB API..."):
                    code, data = alphafold_lookup(uniprot_id.strip())

                if isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    pdb_url = entry.get("pdbUrl") or entry.get("pdb_url")
                    cif_url = entry.get("cifUrl") or entry.get("cif_url")

                    if pdb_url:
                        with st.spinner("Downloading AlphaFold PDB..."):
                            _, structure_bytes = http_get_bytes(pdb_url)
                        structure_ext = "pdb"
                        structure_source = f"AlphaFold DB (UniProt {uniprot_id.strip()}) PDB"
                    elif cif_url:
                        with st.spinner("Downloading AlphaFold mmCIF..."):
                            _, structure_bytes = http_get_bytes(cif_url)
                        structure_ext = "cif"
                        structure_source = f"AlphaFold DB (UniProt {uniprot_id.strip()}) mmCIF"
                    else:
                        st.warning("AlphaFold API response did not include pdbUrl/cifUrl.")
                        st.code(json.dumps(entry, indent=2))
                else:
                    st.warning("No AlphaFold prediction found for this UniProt ID.")
                    st.link_button("Open AlphaFold entry page", f"{ALPHAFOLD_ENTRY}/{uniprot_id.strip()}")

            except urllib.error.HTTPError as e:
                st.error(f"AlphaFold API HTTPError: {e.code} {e.reason}")
            except Exception as ex:
                st.error(f"AlphaFold API Error: {ex}")

        if not structure_bytes:
            st.info(
                "ยังไม่มีโครงสร้างให้แสดง 3D:\n"
                "- ใช้ Option A (ColabFold) เพื่อสร้างโครงสร้างฟรี แล้วอัปโหลดไฟล์ PDB/mmCIF ใน Option C\n"
                "- หรือใช้ Option B ใส่ UniProt ID (ถ้ามีอยู่ใน AlphaFold DB)"
            )
            st.stop()

        st.success(f"Structure ready ✅ ({structure_source})")

        b64 = base64.b64encode(structure_bytes).decode("utf-8")
        ext = structure_ext if structure_ext in ["pdb", "cif"] else "pdb"

        ngl_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js"></script>
  <style>
    body {{ margin:0; }}
    #viewport {{ width: 100%; height: 540px; }}
  </style>
</head>
<body>
  <div id="viewport"></div>
  <script>
    const stage = new NGL.Stage("viewport");
    window.addEventListener("resize", function(){{ stage.handleResize(); }}, false);

    const b64 = "{b64}";
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i=0; i<len; i++) bytes[i] = binary.charCodeAt(i);

    const blob = new Blob([bytes], {{type: "text/plain"}});
    stage.loadFile(blob, {{ ext: "{ext}" }}).then(function(o){{
      o.addRepresentation("cartoon");
      o.autoView();
    }});
  </script>
</body>
</html>
"""
        st.components.v1.html(ngl_html, height=560, scrolling=False)

        st.download_button(
            "Download structure file",
            data=structure_bytes,
            file_name=f"{sample_id or 'structure'}.{ext}",
            mime="chemical/x-pdb" if ext == "pdb" else "chemical/x-cif",
            key="p_struct_dl",
        )

    st.caption("Structure options: AlphaFold DB (if UniProt exists) or generate using ColabFold then upload PDB/mmCIF.")


# =========================================================
# Router
# =========================================================
if app_mode == "HA/NA Sequence Identification Quick-Check":
    module_ha_na()
else:
    module_dna_protein_3d()
