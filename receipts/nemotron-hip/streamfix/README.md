# Nemotron HIP draft review evidence

Review top1596156e; measured productioncc18314d; baseline8649e985. Full identities are in heads.json and each proof. The fixture-only bridge is under fixture/.

DRAFT / NOT READY TO LAND. Critical NEW/41/46 contracts pass both runtime GPU types with zero known findings; four-chip resources pass. This is not complete-stack qualification. Shared models, old-header negative, production O3 finite bytes, host/static proofs and both-rig Nemotron comparisons are included.

ab_partial contains completed alternating drafts-on measurements and exact outputs. It reports a material decode regression: V620 code32.58%/prose31.50% slower, W7800 code36.65%/prose34.11% slower. Full narrow A/B is now complete: all eight condition quartets and 32 holds preserve complete-token hashes and cold cached-token counts are zero. Drafts-off throughput regresses by31.67–51.05%; see ab_complete/summary.md for every rate and cold timing/drift. Intermediate-head checks, integration and additional CONTRIBUTING lifecycle/context/memory/comparator requirements remain pending. Current receipts must not be relabelled as another source or as full qualification.

Historical evidence on the parent receipt commit retains its original identities. No upstream readiness or all-platform claim is made.
