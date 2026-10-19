# Run from this receipt directory, with Zig0.17.0 and the actual reviewed checkout.
# SOURCE_TREE denotes a clean checkout of1596156e21769324b675ef5e0caae0452f361660.
zig test -O ReleaseFast --cache-dir .fixture-cache --global-cache-dir .fixture-global --dep family -Mroot=probe.zig -Mfamily="$SOURCE_TREE/zig/src/core/contract.zig" --test-filter fixture2
zig test -O ReleaseFast --cache-dir .fixture-cache --global-cache-dir .fixture-global --dep family -Mroot=probe_actual_rows.zig -Mfamily="$SOURCE_TREE/zig/src/core/contract.zig" --test-filter fixture2
zig test -O ReleaseFast --cache-dir .fixture-old-cache --global-cache-dir .fixture-global --dep family -Mroot=old_probe.zig -Mfamily="$SOURCE_TREE/zig/src/core/contract.zig" --test-filter original
