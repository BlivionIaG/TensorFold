Candidate correctness receipts. This package qualifies only the explicitly recorded candidate correctness checks.

Measured candidate: 729833615595df66c9f9a438ce9a85061843f010. Baseline: 8649e9858b38e9ef3dcd8d579879c6512a363050. The full-tree bridge records identical source at private review60a8; measurements retain their actual source identity.

Original raw artifacts remain local. manifest.json binds original bytes and redacted copies. Only personal home paths, local scratch paths, the private SSH alias, and the two Mac roots are replaced: the Mac receipt directory → <MAC_RECEIPT>, the isolated candidate tree → <CANDIDATE_TREE>. Failed infrastructure attempts are retained separately; their failures are not numerical failures or successful tests. Binary output dumps are retained locally; comparison JSON records per-fixture hashes. Performance recovery is separate and pending.

Independent audit validated this candidate correctness stage: strict49/49 on gfx1100 and gfx1030, zero failures/known findings/limit problems, all five planted controls red; finite1080/540 cases,60/24 finite-input NaN controls and zero byte mismatches. Four-chip resource records pass; fresh Mac host recipe and plain build pass. Shared model/kernel/Nemotron and candidate Linux follow-up are pending separately; performance is not qualified by this package.
