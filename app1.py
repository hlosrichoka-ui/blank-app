import re
import json
import time
import base64
import urllib.request
import urllib.error

import pandas as pd
import streamlit as st
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="NT→Protein + SWISS-MODEL 3D", layout="centered")
st.title("NT → Protein + Protein schematic + SWISS-MODEL 3D")

SWISSMODEL_BASE = "https://swissmodel.expasy.org"
COREAPI_BASE = f"{SWISSMODEL_BASE}/coreapi"

AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

# =========================================================
# Utils: sequence
# =========================================================
def clean_nt(seq: str) -> str:
    """Remove FASTA headers; keep letters; convert non-ATGC to N (keep length stable)."""
    lines = [l.strip() for l in seq.splitlines() if l.strip() and not l.strip().startswith(">")]
    s = "".join(lines).upper()
    s = re.sub(r"[^A-Z]", "", s)
    s = re.sub(r"[^ATGC]", "N", s)
    return s

def translate_frame(nt: str, strand: str, frame: int) -> str:
    if strand == "-":
        nt = str(Seq(nt).reverse_complement())
    nt = nt[frame:]
    nt = nt[: (len(nt)//3)*3]
    return str(Seq(nt).translate(to_stop=False)) if nt else ""

def best_orf_6frames(nt: str):
    """Pick the longest AA before the first stop among 6 frames."""
    best = {"strand": "+", "frame": 0, "aa_full": "", "aa_orf": "", "orf_len": 0, "stop_count": 10**9}
    for strand in ["+", "-"]:
        for frame in [0, 1, 2]:
            aa_full = translate_frame(nt, strand, frame)
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
# Utils: schematic (no matplotlib)
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
    vals = []
    pos = []
    for i in range(len(aa) - window + 1):
        seg = aa[i:i+window]
        vals.append(sum(kd[x] for x in seg) / window)
        pos.append(i + 1)  # 1-based window start
    return pos, vals

def predict_tm_segments(pos, vals, threshold=1.6, min_len=18):
    """Heuristic TM: consecutive windows above threshold >= min_len."""
    segs = []
    start = None
    for p, v in zip(pos, vals):
        if v >= threshold:
            if start is None:
                start = p
        else:
            if start is not None:
                end = p - 1
                if (end - start + 1) >= min_len:
                    segs.append((start, end))
                start = None
    if start is not None:
        end = pos[-1]
        if (end - start + 1) >= min_len:
            segs.append((start, end))
    return segs

def text_schematic(length_aa: int, tm_segments):
    """
    Make a simple bar where TM segments are '█' and others '─'.
    Note: TM segments are in 'window-start' coordinates; we map roughly onto AA positions.
    """
    if length_aa <= 0:
        return ""
    bar = ["─"] * length_aa
    for (s, e) in tm_segments:
        # map window positions onto AA indices (rough)
        s_aa = max(1, s)
        e_aa = min(length_aa, e)
        for i in range(s_aa-1, e_aa):
            bar[i] = "█"
    return "".join(bar)

# =========================================================
# Utils: HTTP (no requests dependency)
# =========================================================
def http_post_json(url: str, payload: dict, headers: dict | None = None, timeout=60):
    headers = headers or {}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))

def http_get(url: str, headers: dict | None = None, timeout=60):
    headers = headers or {}
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read()

def http_get_json(url: str, headers: dict | None = None, timeout=60):
    code, raw = http_get(url, headers=headers, timeout=timeout)
    return code, json.loads(raw.decode("utf-8"))

# =========================================================
# SWISS-MODEL API helpers
# =========================================================
def swissmodel_login_get_token(username: str, password: str):
    # POST /api-token-auth/
    url = f"{COREAPI_BASE}/api-token-auth/"
    code, data = http_post_json(url, {"username": username, "password": password})
    # expected: {"token": "..."}
    token = data.get("token")
    return code, data, token

def swissmodel_submit_automodel(token: str, target_sequence: str, project_title: str):
    # POST /automodel/
    url = f"{COREAPI_BASE}/automodel/"
    headers = {"Authorization": f"Token {token}"}
    code, data = http_post_json(url, {"target_sequences": target_sequence, "project_title": project_title}, headers=headers)
    return code, data

def swissmodel_get_project(token: str, project_id: str):
    # GET /projects/{project_id}/
    url = f"{COREAPI_BASE}/projects/{project_id}/"
    headers = {"Authorization": f"Token {token}"}
    return http_get_json(url, headers=headers)

def swissmodel_get_project_summary(token: str, project_id: str):
    # GET /project/{project_id}/models/summary/
    url = f"{COREAPI_BASE}/project/{project_id}/models/summary/"
    headers = {"Authorization": f"Token {token}"}
    return http_get_json(url, headers=headers)

def swissmodel_download_model_cif(token: str, project_id: str, model_id: str):
    # GET /project/{project_id}/models/{model_id}.cif
    url = f"{COREAPI_BASE}/project/{project_id}/models/{model_id}.cif"
    headers = {"Authorization": f"Token {token}"}
    code, raw = http_get(url, headers=headers, timeout=120)
    return code, raw

# =========================================================
# UI
# =========================================================
sample_id = st.text_input("Sample ID")
nt_raw = st.text_area("Paste nucleotide sequence (FASTA)", height=220)

mode = st.radio("Translation mode", ["Auto (best ORF)", "Manual"], horizontal=True)
c1, c2 = st.columns(2)
strand = c1.selectbox("Strand", ["+", "-"], disabled=(mode == "Auto (best ORF)"))
frame = c2.selectbox("Frame", [0, 1, 2], disabled=(mode == "Auto (best ORF)"))

st.markdown("---")
st.subheader("Schematic settings")
window = st.slider("Hydropathy window", 7, 31, 19, step=2)
tm_threshold = st.slider("TM threshold (heuristic)", 0.5, 3.0, 1.6, step=0.1)
tm_min_len = st.slider("Min TM length (windows)", 10, 30, 18, step=1)

st.markdown("---")
st.subheader("SWISS-MODEL connection (optional)")

with st.expander("Connect to SWISS-MODEL and show 3D structure"):
    st.caption("Uses SWISS-MODEL REST API (token-based). Your credentials are only used for this session.")
    sm_user = st.text_input("SWISS-MODEL username", value="", help="Your SWISS-MODEL account username")
    sm_pass = st.text_input("SWISS-MODEL password", value="", type="password", help="Used only to obtain an API token")
    project_title = st.text_input("Project title (SWISS-MODEL)", value=f"{sample_id or 'protein'}_automodel")

    poll_every = st.number_input("Polling interval (sec)", value=10, min_value=5, max_value=60, step=5)
    poll_max_rounds = st.number_input("Max polling rounds", value=30, min_value=5, max_value=120, step=5)

# =========================================================
# Run
# =========================================================
if st.button("Translate → Schematic → (Optional) SWISS-MODEL 3D", type="primary"):
    NT = clean_nt(nt_raw)
    if not NT:
        st.error("Invalid nucleotide sequence.")
        st.stop()

    # Translate
    if mode == "Auto (best ORF)":
        best = best_orf_6frames(NT)
        strand_use = best["strand"]
        frame_use = best["frame"]
        aa_orf = best["aa_orf"]
        stop_count = best["stop_count"]
        note = f"Auto ORF: strand {strand_use}, frame {frame_use} | ORF={best['orf_len']} aa | stops(full)={stop_count}"
    else:
        strand_use = strand
        frame_use = int(frame)
        aa_full = translate_frame(NT, strand_use, frame_use)
        aa_orf = aa_full.split("*")[0] if aa_full else ""
        stop_count = aa_full.count("*") if aa_full else 0
        note = f"Manual: strand {strand_use}, frame {frame_use} | ORF={len(aa_orf)} aa | stops(full)={stop_count}"

    protein = sanitize_protein(aa_orf)
    if not protein:
        st.error("Protein translation failed (check frame/strand or too many ambiguities).")
        st.stop()

    # Protein properties
    pa = ProteinAnalysis(protein)
    mw = pa.molecular_weight()
    pi = pa.isoelectric_point()
    gravy = pa.gravy()
    arom = pa.aromaticity()

    st.markdown("## 🔬 Protein result")
    st.write(f"**Sample:** {sample_id or '-'}")
    st.write(f"**Input:** {len(NT)} nt")
    st.write(f"**Translation:** {note}")
    st.write(f"**Protein length:** {len(protein)} aa")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MW (Da)", f"{mw:,.1f}")
    m2.metric("pI", f"{pi:.2f}")
    m3.metric("GRAVY", f"{gravy:.2f}")
    m4.metric("Aromaticity", f"{arom:.3f}")

    st.markdown("### Protein FASTA")
    fasta = f">{sample_id or 'protein'}|strand={strand_use}|frame={frame_use}\n{protein}\n"
    st.code(fasta, language="text")

    # Schematic
    st.markdown("## 🧬 Protein schematic (QC-friendly)")
    pos, vals = kyte_doolittle(protein, window=window)
    tm = predict_tm_segments(pos, vals, threshold=tm_threshold, min_len=tm_min_len)

    if vals:
        df = pd.DataFrame({"Hydropathy (KD)": vals}, index=pos)
        st.line_chart(df)
    else:
        st.info("Protein too short for this hydropathy window.")

    st.write("**TM-like segments (heuristic):**", ", ".join([f"{s}-{e}" for s, e in tm]) if tm else "-")
    st.code(text_schematic(len(protein), tm))

    # -----------------------------------------------------
    # SWISS-MODEL: submit + poll + render
    # -----------------------------------------------------
    do_swiss = bool(sm_user and sm_pass)
    if not do_swiss:
        st.warning("If you want 3D structure: expand SWISS-MODEL section and fill username/password, then run again.")
        st.stop()

    st.markdown("---")
    st.markdown("## 🧩 SWISS-MODEL: submit & fetch model")
    try:
        with st.spinner("Logging in to SWISS-MODEL to obtain API token..."):
            code, login_json, token = swissmodel_login_get_token(sm_user, sm_pass)

        if not token:
            st.error("Login failed (no token returned).")
            st.code(json.dumps(login_json, indent=2))
            st.stop()

        st.success("Token obtained.")

        with st.spinner("Submitting AutoModel job to SWISS-MODEL..."):
            code, submit_json = swissmodel_submit_automodel(token, protein, project_title)

        # usually includes project_id (but keep robust)
        project_id = submit_json.get("project_id") or submit_json.get("id") or submit_json.get("project")
        st.write("**Submit response:**")
        st.code(json.dumps(submit_json, indent=2))

        if not project_id:
            st.error("Could not find project_id in response. Please check the submit response above.")
            st.stop()

        st.success(f"Submitted. Project ID: {project_id}")

        # Poll for completion / models summary
        model_id = None
        last_summary = None

        for i in range(int(poll_max_rounds)):
            with st.spinner(f"Polling SWISS-MODEL... ({i+1}/{int(poll_max_rounds)})"):
                # summary endpoint tends to show models when ready
                scode, summary = swissmodel_get_project_summary(token, str(project_id))
                last_summary = summary

            # Heuristic: find a model_id if present
            # Summary schema can evolve; we try common patterns.
            candidate_ids = []
            if isinstance(summary, dict):
                # some APIs return {"models":[{"model_id":"..."}]}
                models = summary.get("models") or summary.get("results") or summary.get("data") or []
                if isinstance(models, list):
                    for m in models:
                        if isinstance(m, dict):
                            mid = m.get("model_id") or m.get("id") or m.get("modelId")
                            if mid:
                                candidate_ids.append(str(mid))

            if candidate_ids:
                model_id = candidate_ids[0]
                break

            time.sleep(int(poll_every))

        st.markdown("### Project/model status")
        st.code(json.dumps(last_summary, indent=2) if last_summary else "No summary.")

        if not model_id:
            st.warning(
                "Model is not ready yet (or summary format differs). "
                "You can wait and run again, or open SWISS-MODEL workspace to check the project."
            )
            st.write("Open SWISS-MODEL:", f"{SWISSMODEL_BASE}/")
            st.stop()

        st.success(f"Model ready. model_id = {model_id}")

        # Download CIF
        with st.spinner("Downloading model (mmCIF)..."):
            dcode, cif_bytes = swissmodel_download_model_cif(token, str(project_id), str(model_id))

        cif_text = cif_bytes.decode("utf-8", errors="replace")

        st.download_button(
            "Download model (mmCIF)",
            data=cif_bytes,
            file_name=f"{sample_id or 'protein'}_swissmodel_{project_id}_{model_id}.cif",
            mime="chemical/x-cif",
        )

        # Render 3D with NGL (client-side)
        st.markdown("## 🧊 3D Structure Viewer")
        cif_b64 = base64.b64encode(cif_bytes).decode("utf-8")

        ngl_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://unpkg.com/ngl@2.0.0-dev.39/dist/ngl.js"></script>
  <style>
    body {{ margin:0; }}
    #viewport {{ width: 100%; height: 520px; }}
  </style>
</head>
<body>
  <div id="viewport"></div>
  <script>
    const stage = new NGL.Stage("viewport");
    window.addEventListener("resize", function(){{ stage.handleResize(); }}, false);

    // Decode base64 → ArrayBuffer
    const b64 = "{cif_b64}";
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i=0; i<len; i++) bytes[i] = binary.charCodeAt(i);

    const blob = new Blob([bytes], {{type: "text/plain"}});
    stage.loadFile(blob, {{ ext: "cif" }}).then(function(o){{
      o.addRepresentation("cartoon");
      o.autoView();
    }});
  </script>
</body>
</html>
"""
        st.components.v1.html(ngl_html, height=540, scrolling=False)

    except urllib.error.HTTPError as e:
        st.error(f"HTTPError: {e.code} {e.reason}")
        try:
            body = e.read().decode("utf-8", errors="replace")
            st.code(body)
        except Exception:
            pass
    except Exception as ex:
        st.error(f"Error: {ex}")

st.caption(
    "SWISS-MODEL AutoModel uses homology modeling; output quality depends on available templates. "
    "API endpoints: api-token-auth + automodel + project models. "
)
