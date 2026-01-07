def query_centric_local_compare(ref: str, qry: str):
    """
    Local alignment, QUERY-centric metrics.

    Returns:
      identity_input (%): matches / len(query)
      identity_aligned (%): matches / aligned_query_bases
      coverage_input (%): aligned_query_bases / len(query)
      mismatch_positions_abs: 1-based positions on query where aligned but mismatch
      uncovered_positions_abs: 1-based positions on query not aligned at all
      matches: number of matches in aligned region
      aligned_query_bases: number of query bases aligned
    """
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(ref, qry)))

    # 🔧 FIX HERE
    q_blocks = aln.aligned[1]  # numpy array in some versions
    total_query = len(qry)

    if q_blocks is None or len(q_blocks) == 0:
        # no alignment at all
        uncovered = list(range(1, total_query + 1))
        return 0.0, 0.0, 0.0, [], uncovered, 0, 0

    # Covered query positions (1-based)
    covered = set()
    for qs, qe in q_blocks:
        for p in range(qs + 1, qe + 1):
            covered.add(p)

    uncovered = [p for p in range(1, total_query + 1) if p not in covered]

    # Alignment strings for mismatch detection
    lines = aln.format().splitlines()
    ref_aln = lines[0].replace(" ", "")
    qry_aln = lines[2].replace(" ", "")

    covered_sorted = sorted(covered)
    covered_idx = 0

    matches = 0
    aligned_query_bases = 0
    mismatch_abs = []

    for r, q in zip(ref_aln, qry_aln):
        if q == "-":
            continue
        if covered_idx >= len(covered_sorted):
            break

        q_abs = covered_sorted[covered_idx]
        covered_idx += 1

        aligned_query_bases += 1
        if r == q:
            matches += 1
        else:
            mismatch_abs.append(q_abs)

    identity_input = (matches / total_query * 100) if total_query else 0.0
    identity_aligned = (matches / aligned_query_bases * 100) if aligned_query_bases else 0.0
    coverage_input = (aligned_query_bases / total_query * 100) if total_query else 0.0

    return (
        identity_input,
        identity_aligned,
        coverage_input,
        mismatch_abs,
        uncovered,
        matches,
        aligned_query_bases,
    )
