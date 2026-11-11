#!/bin/sh
# Fill the @NAME@ placeholders of one machine's service files from a site file (see site.conf.example).
# usage: render.sh TARGET SITE_CONF OUT_DIR    TARGET: head (rank 0 + listen daemon) | worker (rank 1) | mac (decoder)
# Writes OUT_DIR/{systemd,etc,libexec,launchd}/..., the layout the guide installs from. Refuses a placeholder that
# the site file leaves unset, so an unfilled file never reaches a machine.
set -eu
[ $# -eq 3 ] || { sed -n '2,4p' "$0" >&2; exit 2; }
target=$1 conf=$2 out=$3
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$conf" ] || { echo "no site file $conf" >&2; exit 2; }

# KEY=value lines; the value is the rest of the line, verbatim. Escape for the sed replacement.
subst=$(mktemp)
trap 'rm -f "$subst"' EXIT
while IFS= read -r line || [ -n "$line" ]; do
    case $line in '' | '#'*) continue ;; esac
    key=${line%%=*}
    case $key in *[!A-Z0-9_]* | '') echo "bad line in $conf: $line" >&2; exit 2 ;; esac
    value=${line#*=}
    escaped=$(printf '%s' "$value" | sed -e 's/[\\|&]/\\&/g')
    printf 's|@%s@|%s|g\n' "$key" "$escaped" >>"$subst"
done <"$conf"

emit() { # SOURCE DESTINATION MODE
    mkdir -p "$(dirname "$out/$2")"
    sed -f "$subst" "$here/$1" >"$out/$2"
    chmod "$3" "$out/$2"
    if left=$(grep -on '@[A-Z][A-Z0-9_]*@' "$out/$2"); then
        echo "$2: not set in $conf:" >&2
        echo "$left" | sed 's/^/  /' >&2
        exit 1
    fi
}
link=$(sed -n 's/^LINK_NAME=//p' "$conf")
[ -n "$link" ] || { echo "LINK_NAME is not set in $conf" >&2; exit 2; }

case $target in
    head)
        emit 'systemd/mcdma-rpcd-listen@.service' 'systemd/mcdma-rpcd-listen@.service' 644
        emit systemd/tf-pd-prefill-rank0.service systemd/tf-pd-prefill-rank0.service 644
        emit systemd/mcdma-listen.env "etc/mcdma-listen-$link.env" 644
        ;;
    worker) emit systemd/tf-pd-prefill-rank1.service systemd/tf-pd-prefill-rank1.service 644 ;;
    mac)
        for plist in restore-links connect decoder; do
            emit "macos/dev.tensorfold.pd-$plist.plist" "launchd/dev.tensorfold.pd-$plist.plist" 644
        done
        emit macos/bin/tf-pd-restore-links libexec/tf-pd-restore-links 755
        emit macos/bin/tf-pd-decoder-run libexec/tf-pd-decoder-run 755
        emit bin/tf-pd-health libexec/tf-pd-health 755
        exit 0
        ;;
    *) echo "TARGET is head, worker or mac" >&2; exit 2 ;;
esac
# every pair machine: the Mac-facing port, the tuned profile, the helpers
emit systemd/tf-pd-link-up.service systemd/tf-pd-link-up.service 644
emit systemd/mac-link.env etc/mac-link.env 644
emit systemd/prefill.env etc/prefill.env 644
emit bin/tf-pd-link-up libexec/tf-pd-link-up 755
emit bin/tf-pd-health libexec/tf-pd-health 755
