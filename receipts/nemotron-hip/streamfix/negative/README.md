# Actual old-header negative control

SUMMARY gfx1100 49 kernels: 30 pass gates 1-4, 19 fail, 0 known findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems; gates 132324 ms, limits 3894 ms

The 19 real nonfinite failures are 18 plain stream width/group cases and the single relu2 case. The paired fixture passes in this seeded negative run; no paired failure is claimed. All five planted controls are red, zero known findings and zero limit problems.

The measured source is 5f453065, rebuilt with its own actual-old metadata. The final candidate cc18314d is a distinct source. Source reconstruction and full IDs are in reproduce_negative_tree.json. Apply the patch in a clean checkout of that final source and verify git write-tree against the recorded hash.

The initial stale-resource dump is retained separately and is not the qualified numerical evidence. Final resource audit succeeded before the strict GPU run. Published logs only normalize workstation and temporary paths; audit_report.json records both raw and published SHA256. Complete untouched raw logs remain local. No binaries are included.
