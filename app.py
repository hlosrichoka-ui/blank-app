import re
import streamlit as st

# =========================================================
# References (user-provided)
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

# =========================================================
# Utility
# =========================================================
def clean_nt(seq: str) -> str:
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    joined = "".join(lines)
    # keep only ATGC (drop ambiguity)
    return re.sub(r"[^ATGCatgc]", "", joined).upper()

def revcomp(seq: str) -> str:
    comp = str.maketrans("ATGC", "TACG")
    return seq.translate(comp)[::-1]

def smith_waterman_local(ref: str, qry: str, match=2, mismatch=-1, gap=-3):
    """
    Smith–Waterman local alignment (simple gap penalty, BLAST-like HSP).
    Returns dict with:
      aligned_ref, aligned_qry,
      ref_start, ref_end, qry_start, qry_end (1-based inclusive),
      matches, mismatches, gaps, aln_len,
      score,
      mismatch_positions_on_query (1-based absolute),
      covered_positions_on_query (set of 1-based absolute positions)
    """
    n = len(ref)
    m = len(qry)
    if n == 0 or m == 0:
        return None

    # DP matrix and traceback: 0=stop, 1=diag, 2=up (gap in qry), 3=left (gap in ref)
    H = [[0]*(m+1) for _ in range(n+1)]
    T = [[0]*(m+1) for _ in range(n+1)]

    best_score = 0
    best_pos = (0, 0)

    for i in range(1, n+1):
        ri = ref[i-1]
        for j in range(1, m+1):
            qj = qry[j-1]
            diag = H[i-1][j-1] + (match if ri == qj else mismatch)
            up   = H[i-1][j] + gap
            left = H[i][j-1] + gap
            val = max(0, diag, up, left)
            H[i][j] = val
            if val == 0:
                T[i][j] = 0
            elif val == diag:
                T[i][j] = 1
            elif val == up:
                T[i][j] = 2
            else:
                T[i][j] = 3

            if val > best_score:
                best_score = val
                best_pos = (i, j)

    if best_score == 0:
        return None

    # Traceback from best_pos
    i, j = best_pos
    aligned_ref = []
    aligned_qry = []

    matches = 0
    mismatches = 0
    gaps = 0

    # Track query absolute positions covered/mismatched
    mismatch_positions = []
    covered_positions = set()

    # End positions (1-based inclusive) of local alignment in original strings
    ref_end = i
    qry_end = j

    while i > 0 and j > 0 and T[i][j] != 0:
        tb = T[i][j]
        if tb == 1:  # diag
            r = ref[i-1]
            q = qry[j-1]
            aligned_ref.append(r)
            aligned_qry.append(q)
            if r == q:
                matches += 1
            else:
                mismatches += 1
                mismatch_positions.append(j)  # query position (1-based, current j)
            covered_positions.add(j)
            i -= 1
            j -= 1
        elif tb == 2:  # up: gap in query
            aligned_ref.append(ref[i-1])
            aligned_qry.append("-")
            gaps += 1
            i -= 1
        else:  # left: gap in ref
            aligned_ref.append("-")
            aligned_qry.append(qry[j-1])
            gaps += 1
            # query base is present, considered covered but not compared to a ref base
            covered_positions.add(j)
            j -= 1

    ref_start = i + 1
    qry_start = j + 1

    aligned_ref = "".join(reversed(aligned_ref))
    aligned_qry = "".join(reversed(aligned_qry))
    aln_len = len(aligned_ref)

    # mismatch_positions were collected in reverse during traceback; normalize sort
    mismatch_positions = sorted(set(mismatch_positions))

    return {
        "aligned_ref": aligned_ref,
        "aligned_qry": aligned_qry,
        "ref_start": ref_start,
        "ref_end": ref_end,
        "qry_start": qry_start,
        "qry_end": qry_end,
        "matches": matches,
        "mismatches": mismatches,
        "gaps": gaps,
        "aln_len": aln_len,
        "score": best_score,
        "mismatch_positions_on_query": mismatch_positions,
        "covered_positions_on_query": covered_positions,
    }

def summarize_alignment(ref: str, qry: str):
    """Run local alignment and compute BLAST-like + query-centric metrics."""
    aln = smith_waterman_local(ref, qry)
    if aln is None:
        total_q = len(qry)
        return {
            "ok": False,
            "identity_alnlen_pct": 0.0,
            "identities_X": 0,
            "aln_len_Y": 0,
            "gaps_g": 0,
            "gaps_pct": 0.0,
            "coverage_input_pct": 0.0,
            "identity_input_pct": 0.0,
            "qry_range": "-",
            "ref_range": "-",
            "mismatch_positions": [],
            "uncovered_positions": list(range(1, total_q+1)),
            "aligned_ref": "",
            "aligned_qry": "",
        }

    X = aln["matches"]
    Y = aln["aln_len"]
    g = aln["gaps"]

    identity_alnlen_pct = (X / Y * 100) if Y else 0.0
    gaps_pct = (g / Y * 100) if Y else 0.0

    covered = aln["covered_positions_on_query"]
    total_q = len(qry)
    coverage_input_pct = (len(covered) / total_q * 100) if total_q else 0.0
    identity_input_pct = (X / total_q * 100) if total_q else 0.0

    uncovered_positions = [p for p in range(1, total_q+1) if p not in covered]

    return {
        "ok": True,
        "identity_alnlen_pct": identity_alnlen_pct,
        "identities_X": X,
        "aln_len_Y": Y,
        "gaps_g": g,
        "gaps_pct": gaps_pct,
        "coverage_input_pct": coverage_input_pct,
        "identity_input_pct": identity_input_pct,
        "qry_range": f'{aln["qry_start"]}-{aln["qry_end"]}',
        "ref_range": f'{aln["ref_start"]}-{aln["ref_end"]}',
        "mismatch_positions": aln["mismatch_positions_on_query"],
        "uncovered_positions": uncovered_positions,
        "aligned_ref": aln["aligned_ref"],
        "aligned_qry": aln["aligned_qry"],
        "score": aln["score"],
    }

def to_csv_text(rows: list) -> str:
    # minimal CSV writer (no pandas)
    if not rows:
        return ""
    cols = list(rows[0].keys())
    out = [",".join(cols)]
    for r in rows:
        line = []
        for c in cols:
            v = r.get(c, "")
            s = str(v).replace('"', '""')
            if "," in s or "\n" in s or '"' in s:
                s = f'"{s}"'
            line.append(s)
        out.append(",".join(line))
    return "\n".join(out)

# =========================================================
# UI
# =========================================================
st.caption("Local alignment (Smith–Waterman) แบบ BLAST-like: ไม่ต้องเริ่มจากเบสแรก และเทียบทั้ง sense/antisense")

sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)
query_raw = st.text_area("Paste Nucleotide Sequence (FASTA or raw):", height=220)

IDENTITY_ALNLEN_PASS = st.number_input("PASS threshold: BLAST-like Identity (% X/Y)", value=99.0, step=0.1)
COVERAGE_PASS = st.number_input("PASS threshold: Coverage (% of input aligned)", value=90.0, step=1.0)

show_alignment_strings = st.checkbox("Show aligned strings (Ref/Query)", value=False)
show_positions = st.checkbox("Show mismatch/uncovered positions", value=True)

# Prepare refs
HA_REF = clean_nt(HA_REF_RAW)
NA_REF = clean_nt(NA_REF_RAW)

if st.button("Analyze Sequence", type="primary"):
    Q = clean_nt(query_raw)
    if not Q:
        st.error("Please paste a valid DNA sequence containing A/T/G/C.")
        st.stop()

    Q_rc = revcomp(Q)

    # Align HA sense/antisense
    ha_sense = summarize_alignment(HA_REF, Q)
    ha_anti  = summarize_alignment(HA_REF, Q_rc)
    # pick best by identity_alnlen_pct then alignment length
    ha_best = ha_sense if (ha_sense["identity_alnlen_pct"], ha_sense["aln_len_Y"]) >= (ha_anti["identity_alnlen_pct"], ha_anti["aln_len_Y"]) else ha_anti
    ha_strand = "sense" if ha_best is ha_sense else "antisense (reverse-complement)"

    # Align NA sense/antisense
    na_sense = summarize_alignment(NA_REF, Q)
    na_anti  = summarize_alignment(NA_REF, Q_rc)
    na_best = na_sense if (na_sense["identity_alnlen_pct"], na_sense["aln_len_Y"]) >= (na_anti["identity_alnlen_pct"], na_anti["aln_len_Y"]) else na_anti
    na_strand = "sense" if na_best is na_sense else "antisense (reverse-complement)"

    # Choose HA vs NA (BLAST-like)
    if (ha_best["identity_alnlen_pct"], ha_best["aln_len_Y"]) >= (na_best["identity_alnlen_pct"], na_best["aln_len_Y"]):
        gene_identified = "HA"
        gene_label = HA_REF_LABEL
        strand_used = ha_strand
        res = ha_best
    else:
        gene_identified = "NA"
        gene_label = NA_REF_LABEL
        strand_used = na_strand
        res = na_best

    # QC assessment
    if gene_identified != expected_gene:
        qc = "❌ FAIL (Gene mismatch)"
        qc_flag = "FAIL"
    elif res["identity_alnlen_pct"] < IDENTITY_ALNLEN_PASS or res["coverage_input_pct"] < COVERAGE_PASS:
        qc = "⚠️ INVESTIGATE (Low BLAST-like identity / coverage)"
        qc_flag = "INVESTIGATE"
    else:
        qc = "✅ PASS"
        qc_flag = "PASS"

    # =====================================================
    # Display
    # =====================================================
    st.markdown("---")
    st.subheader("🔍 Analysis Result (BLAST-like)")
    st.write(f"**Sample ID:** {sample_id if sample_id else '-'}")
    st.write(f"**Organism:** {ORGANISM_LABEL}")
    st.write(f"**Gene Identified:** {gene_identified} — ({gene_label})")
    st.write(f"**Subtype:** {SUBTYPE_LABEL}")
    st.write(f"**Strand used:** {strand_used}")

    st.write(f"**Identities:** {res['identities_X']}/{res['aln_len_Y']} ({res['identity_alnlen_pct']:.2f}%)")
    st.write(f"**Gaps:** {res['gaps_g']}/{res['aln_len_Y']} ({res['gaps_pct']:.2f}%)")
    st.write(f"**Alignment ranges:**  Query {res['qry_range']}  |  Ref {res['ref_range']}")

    st.markdown("#### QC-friendly (query-centric)")
    st.write(f"**Identity (% of input):** {res['identity_input_pct']:.2f}%")
    st.write(f"**Coverage (% of input aligned):** {res['coverage_input_pct']:.2f}%")

    st.markdown(f"### QC Assessment: {qc}")

    if show_positions:
        with st.expander("Mismatch / Uncovered positions (1-based on INPUT)"):
            st.write(f"Mismatch positions (aligned but different): {len(res['mismatch_positions'])}")
            st.code(", ".join(map(str, res["mismatch_positions"])) if res["mismatch_positions"] else "None")

            st.write(f"Uncovered positions (not in aligned region): {len(res['uncovered_positions'])}")
            preview = res["uncovered_positions"][:200]
            tail = "" if len(res["uncovered_positions"]) <= 200 else f"\n... (+{len(res['uncovered_positions'])-200} more)"
            st.code(", ".join(map(str, preview)) + tail if preview else "None")

    if show_alignment_strings and res["ok"]:
        with st.expander("Aligned strings (Ref/Query)"):
            st.code(res["aligned_ref"], language="text")
            st.code(res["aligned_qry"], language="text")

    # =====================================================
    # Download CSV
    # =====================================================
    row = {
        "Sample ID": sample_id,
        "Expected Gene": expected_gene,
        "Gene Identified": gene_identified,
        "Subtype": SUBTYPE_LABEL,
        "Strand used": strand_used,
        "Identities X": res["identities_X"],
        "Alignment length Y": res["aln_len_Y"],
        "BLAST-like Identity (% X/Y)": round(res["identity_alnlen_pct"], 2),
        "Gaps g": res["gaps_g"],
        "Gaps (% g/Y)": round(res["gaps_pct"], 2),
        "Identity (% of input)": round(res["identity_input_pct"], 2),
        "Coverage (% of input aligned)": round(res["coverage_input_pct"], 2),
        "Query aligned range (1-based)": res["qry_range"],
        "Ref aligned range (1-based)": res["ref_range"],
        "Mismatch positions (1-based query)": ";".join(map(str, res["mismatch_positions"])) if res["mismatch_positions"] else "",
        "Uncovered positions (1-based query)": ";".join(map(str, res["uncovered_positions"])) if res["uncovered_positions"] else "",
        "Reference used": gene_label,
        "QC Flag": qc_flag,
        "QC Assessment": qc,
        "Alignment mode": "local (Smith–Waterman)",
    }

    csv_text = to_csv_text([row])
    st.download_button(
        "Download QC Report (CSV)",
        data=csv_text.encode("utf-8"),
        file_name=f"{sample_id or 'qc_report'}_HA_NA_blastlike.csv",
        mime="text/csv",
    )
