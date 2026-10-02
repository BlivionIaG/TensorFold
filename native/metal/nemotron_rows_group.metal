
  // One threadgroup of T >= E threads: thread e counts the pairs that picked expert e; the used experts, in
  // increasing id, get groups u = 0, 1, ...: UIDS[u] = e, START[u] / COUNT[u] = its run in MEMBERS, where its
  // pairs sit in increasing order; UCOUNT[0] = the number of groups.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int P = pairs[0];
  threadgroup uint ids[MAXP];
  threadgroup int sg_pairs[T / 32], sg_used[T / 32];
  for (int p = int(t); p < P; p += T) ids[p] = IDS[p];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = int(t);
  int count = 0;
  if (e < E)
    for (int p = 0; p < P; p++) count += int(ids[p]) == e ? 1 : 0;
  const int used = count > 0 ? 1 : 0;
  const int pairs_before = simd_prefix_exclusive_sum(count);
  const int used_before = simd_prefix_exclusive_sum(used);
  if (lane == 31) { sg_pairs[sg] = pairs_before + count; sg_used[sg] = used_before + used; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int start = pairs_before, u = used_before;
  for (uint q = 0; q < sg; q++) { start += sg_pairs[q]; u += sg_used[q]; }
  if (used) {
    UIDS[u] = uint(e);
    START[u] = start;
    COUNT[u] = count;
    int m = start;
    for (int p = 0; p < P; p++)
      if (int(ids[p]) == e) MEMBERS[m++] = p;
  }
  if (int(t) == T - 1) UCOUNT[0] = u + used;
