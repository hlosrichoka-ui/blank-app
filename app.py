import time
import re
import requests
import streamlit as st
from Bio.Seq import Seq

EBI_BASE = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"

st.set_page_config(page_title="HA/NA Quick-Check (EBI BLAST)")
st.title("HA/NA Sequence Identity Quick-Check (QC) — EBI BLAST")

# --- Inputs ---
email = st.text_input("Email (required by EBI fair-use policy)", placeholder="your.name@org.com")
sample_id = st.text_input("Sample ID")
expected_gene = st.radio("Expected Gene", ["HA", "NA"], horizontal=True)

blast_program = st.selectbox(
    "BLAST program",
    ["blastn", "tblastx", "blastx", "tblastn", "blastp"],
    index=0
)

# For nucleotide HA/NA quick-check, ENA coding sequences is a reasonable start.
# (You can change to other EM_* or UniProt options later.)
database = st.selectbox(
    "Database",
    [
        "EM_CDS",     # ENA coding sequences
        "EM_NCS",     # ENA non-coding
        "EM_RRNA",    # ENA rRNA
        "UNIPROT",    # UniProtKB
        "SP",         # SwissProt
        "TR",         # TrEMBL
        "UniVec"      # vectors
    ],
    index=0
)

sequence = st.text_area(
    "Paste nucleotide/protein sequence (FASTA or raw):",
    height=220
)

max_wait_seconds = st.slider("Max wait for BLAST (seconds)", 20, 180, 60, 10)

# --- Helpers ---
def clean_sequence(seq: str) -> str:
    # keep only letters + fasta header stripping
    seq = re.sub(r">.*\n", "", seq)
    return re.sub(r"[^A-Za-z]", "", seq).upper()

def ebi_post_run(email: str, program: str, database: str, stype: str, seq: str) -> str:
    # EBI REST: POST /run with form fields
    r = requests.post(
        f"{EBI_BASE}/run",
        data={
            "email": email,
            "program": program,
            "database": database,
            "stype": stype,
            "sequence": seq,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.text.strip()  # jobId

def ebi_get_status(jobid: str) -> str:
    r = requests.get(f"{EBI_BASE}/status/{jobid}", timeout=30)
    r.raise_for_status()
    return r.text.strip()

def ebi_get_result(jobid: str, rtype: str) -> str:
    r = requests.get(f"{EBI_BASE}/result/{jobid}/{rtype}", timeout=120)
    r.raise_for_status()
    return r.text

def guess_gene_from_blast_out(blast_out: str) -> str:
    u = blast_out.upper()
    if "HEMAGGLUTININ" in u or " HA " in u:
        return "HA"
    if "NEURAMINIDASE" in u or " NA " in u:
        return "NA"
    return "Unknown"

def qc_orf_check_if_nt(nt_seq: str) -> str:
    # very rough ORF sanity check: translate 3 frames and see if many stops
    # For QC quick-check only (NOT a validation claim)
    stops = []
    for frame in [0, 1, 2]:
        aa = str(Seq(nt_seq[frame:]).translate(to_stop=False))
        stops.append(aa.count("*"))
    return f"Frame stop counts: {stops} (lower is better)"

# --- UI ---
if st.button("Run EBI BLAST and Analyze"):
    if not email:
        st.error("Please provide email (required by EBI).")
        st.stop()
    if not sequence.strip():
        st.error("Please paste a sequence.")
        st.stop()

    clean_seq = clean_sequence(sequence)
    if len(clean_seq) < 50:
        st.warning("Sequence looks very short (<50 bp/aa). BLAST may be uninformative.")

    # Guess sequence type for EBI 'stype'
    # blastn usually expects dna; blastp expects protein.
    if blast_program == "blastp":
        stype = "protein"
    else:
        stype = "dna"

    st.info(f"Submitting job to EBI BLAST: program={blast_program}, database={database}, stype={stype}")

    try:
        jobid = ebi_post_run(email=email, program=blast_program, database=database, stype=stype, seq=clean_seq)
    except Exception as e:
        st.error(f"Submit failed: {e}")
        st.stop()

    st.success(f"Job submitted. JobID: {jobid}")

    # Poll status
    with st.spinner("Waiting for EBI BLAST to finish..."):
        start = time.time()
        status = "RUNNING"
        while time.time() - start < max_wait_seconds:
            try:
                status = ebi_get_status(jobid)
            except Exception as e:
                st.warning(f"Status check issue: {e}")
                time.sleep(3)
                continue

            if status in ["FINISHED", "ERROR", "FAILURE", "NOT_FOUND"]:
                break
            time.sleep(3)

    st.write(f"Job status: **{status}**")

    if status != "FINISHED":
        st.error("BLAST did not finish within the wait time (or returned an error).")
        st.caption("Tip: increase max wait time, or try again later. Results are typically stored for limited time on the service.")
        st.stop()

    # Fetch results
    try:
        out_txt = ebi_get_result(jobid, "out")   # main BLAST text output
    except Exception as e:
        st.error(f"Failed to fetch BLAST output (out): {e}")
        st.stop()

    # Optional: fetch hit IDs
    ids_txt = ""
    try:
        ids_txt = ebi_get_result(jobid, "ids")
    except Exception:
        pass

    st.subheader("🔍 Quick QC Summary")
    st.write(f"Sample ID: **{sample_id or '-'}**")
    st.write(f"Expected gene: **{expected_gene}**")
    if stype == "dna":
        st.write(qc_orf_check_if_nt(clean_seq))

    gene_guess = guess_gene_from_blast_out(out_txt)
    st.write(f"Gene identified (heuristic from BLAST text): **{gene_guess}**")

    if gene_guess != "Unknown" and gene_guess != expected_gene:
        st.error("QC Assessment: ❌ FAIL (Gene mismatch vs expected)")
    else:
        st.success("QC Assessment: ✅ OK (No obvious gene mismatch from BLAST text)")

    if ids_txt:
        st.markdown("### Top hit IDs (from EBI)")
        st.code("\n".join(ids_txt.strip().splitlines()[:10]))

    st.markdown("### BLAST Output (text)")
    st.text_area("EBI BLAST out", out_txt, height=350)

    st.caption(
        "Notes: This app uses EMBL-EBI Job Dispatcher (NCBI BLAST+) REST endpoints. "
        "Provide a valid email and avoid high-volume submissions per fair-use."
    )
