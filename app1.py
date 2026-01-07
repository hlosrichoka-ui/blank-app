import re
import json
import base64
import urllib.request
import urllib.error

import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="DNA→RNA→Protein + Free 3D Viewer", layout="centered")
st.title("DNA → RNA → Protein + Free Structure Options")

# =========================================================
# Constants / Links (free services)
# =========================================================
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
ALPHAFOLD_ENTRY = "https://www.alphafold.ebi.ac.uk/entry"
COLABFOLD_WEB = "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"

AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

# =========================================================
# Helpers: sequence cleaning & conversion
# =========================================================
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

def best_orf_6frames(dna: str):
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

# =========================================================
# Helpers: simple QC-friendly plots without matplotlib
# =========================================================
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

# =========================================================
# Helpers: HTTP
# =========================================================
def http_get_json(url: str, timeout=60):
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))

def http_get_bytes(url: str, timeout=120):
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read()

# =========================================================
# Helpers: AlphaFold DB API
# =========================================================
def alphafold_lookup(uniprot_id: str):
    """
    AlphaFold DB prediction API:
    GET https://alphafold.ebi.ac.uk/api/prediction/<UniProt>
    Returns list with model file URLs (PDB/mmCIF) when available.
    """
    url = f"{ALPHAFOLD_API}/{uniprot_id}"
    return http_get_json(url)

# =========================================================
# UI
# =========================================================
sample_id = st.text_input("Sample ID")
dna_raw = st.text_area("Paste DNA sequence (FASTA / plain)", height=220)

mode = st.radio("Translation mode", ["Auto (best ORF)", "Manual"], horizontal=True)
c1, c2 = st.columns(2)
strand = c1.selectbox("Strand", ["+", "-"], disabled=(mode == "Auto (best ORF)"))
frame = c2.selectbox("Frame", [0, 1, 2], disabled=(mode == "Auto (best ORF)"))

window = st.slider("Hydropathy window", 7, 31, 19, step=2)

st.markdown("---")
st.subheader("Free structure options")

with st.expander("Option A (FREE): ColabFold / AlphaFold2 (run outside)"):
    st.write("ใช้ ColabFold (ฟรี) เพื่อทำนายโครงสร้างจาก protein sequence แล้วดาวน์โหลดไฟล์ PDB/mmCIF กลับมาอัปโหลดที่ Option C")
    st.link_button("Open ColabFold (AlphaFold2 notebook)", COLABFOLD_WEB)

with st.expander("Option B (FREE if exists): AlphaFold DB by UniProt ID"):
    uniprot_id = st.text_input("UniProt Accession (e.g., Q8AVJ1)", value="")
    st.caption("ถ้ามีโครงสร้างใน AlphaFold DB จะดึงไฟล์ PDB มาแสดง 3D ได้ทันที :contentReference[oaicite:2]{index=2}")

with st.expander("Option C: Upload a structure file (PDB/mmCIF) to view in 3D"):
    uploaded = st.file_uploader("Upload .pdb or .cif", type=["pdb", "cif"])

# =========================================================
# Run main analysis
# =========================================================
if st.button("Convert & Analyze", type="primary"):
    DNA = clean_dna_keep_len(dna_raw)
    if not DNA:
        st.error("Invalid DNA sequence.")
        st.stop()

    RNA = dna_to_rna(DNA)

    # Translate
    if mode == "Auto (best ORF)":
        best = best_orf_6frames(DNA)
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
    st.markdown("## 📈 Protein visualization (free, in-app)")
    pos, vals = kyte_doolittle(protein, window=window)
    if vals:
        df = pd.DataFrame({"Hydropathy (Kyte-Doolittle)": vals}, index=pos)
        st.line_chart(df)
    else:
        st.info("Protein too short for this hydropathy window.")

    # Downloads (DNA/RNA/Protein)
    st.download_button("Download DNA (txt)", DNA.encode("utf-8"), file_name=f"{sample_id or 'seq'}_DNA.txt")
    st.download_button("Download RNA (txt)", RNA.encode("utf-8"), file_name=f"{sample_id or 'seq'}_RNA.txt")
    st.download_button("Download Protein FASTA", fasta.encode("utf-8"), file_name=f"{sample_id or 'protein'}_protein.fasta")

    # =====================================================
    # Structure: Option B (AlphaFold DB)
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
                # pick first model, prefer PDB if present
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

    # Render 3D with NGL (embedded)
    # NGL can load PDB/mmCIF files (supports common formats) :contentReference[oaicite:3]{index=3}
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
    )

st.caption(
    "Free structure routes: AlphaFold DB (if UniProt entry exists) :contentReference[oaicite:4]{index=4} "
    "or ColabFold (AlphaFold2 notebook) to generate a model for new sequences :contentReference[oaicite:5]{index=5}. "
    "3D rendering uses NGL, which supports PDB/mmCIF :contentReference[oaicite:6]{index=6}."
)
