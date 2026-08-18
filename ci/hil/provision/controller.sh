#!/bin/sh
set -eu
set -f

fail() {
    printf 'controller provision: %s\n' "$*" >&2
    exit 1
}

[ "$(id -u)" -eq 0 ] || fail "must run as root"
source_root=${1:-}
case $source_root in /*) ;; *) fail "source root must be absolute" ;; esac
[ -f "$source_root/ci/hil/lab_control.py" ] || fail "invalid source root"
id actions >/dev/null 2>&1 || fail "actions user does not exist"
python3 -c 'import jsonschema, yaml' >/dev/null 2>&1 || \
    fail "python3 jsonschema and PyYAML are required"

version=$(python3 "$source_root/ci/hil/lab_control.py" --version)
case $version in ''|*[!0-9.]*) fail "invalid lab-control version" ;; esac
base=/opt/hil/control
releases=$base/releases
release=$releases/$version
stage=$releases/.stage-$version-$$
case $stage in /opt/hil/control/releases/.stage-*) ;; *) fail "unsafe stage path" ;; esac
[ ! -e "$stage" ] && [ ! -L "$stage" ] || fail "stage path already exists"

install -d -m 0755 -o root -g root "$releases"
install -d -m 0750 -o actions -g actions /opt/hil/runs
install -d -m 0755 -o root -g root /etc/hil /usr/local/bin
install -d -m 0755 -o root -g root "$stage/ci/hil/schemas"
trap 'rm -rf "$stage"' EXIT HUP INT TERM
install -m 0755 -o root -g root \
    "$source_root/ci/hil/lab_control.py" "$stage/ci/hil/lab_control.py"
install -m 0755 -o root -g root \
    "$source_root/ci/hil/capture_uart.py" "$stage/ci/hil/capture_uart.py"
install -m 0644 -o root -g root \
    "$source_root/ci/hil/validate_config.py" "$stage/ci/hil/validate_config.py"
schema_list=$stage/.schema-list
find "$source_root/ci/hil/schemas" -maxdepth 1 -type f -name '*.json' -print | \
    LC_ALL=C sort >"$schema_list"
[ -s "$schema_list" ] || fail "no schemas found"
while IFS= read -r schema; do
    case $schema in
        "$source_root"/ci/hil/schemas/[A-Za-z0-9._-]*.json) ;;
        *) fail "unsafe schema path" ;;
    esac
    install -m 0644 -o root -g root \
        "$schema" "$stage/ci/hil/schemas/$(basename "$schema")"
done <"$schema_list"
rm -f "$schema_list"
(
    cd "$stage"
    find ci -type f -print | LC_ALL=C sort | while IFS= read -r path; do
        sha256sum "$path"
    done >INSTALL-SHA256SUMS
)
chmod 0644 "$stage/INSTALL-SHA256SUMS"

if [ -e "$release" ] || [ -L "$release" ]; then
    [ -d "$release" ] && [ ! -L "$release" ] || fail "release path is unsafe"
    (cd "$release" && sha256sum -c INSTALL-SHA256SUMS >/dev/null) || \
        fail "installed release failed self-verification"
    cmp -s "$stage/INSTALL-SHA256SUMS" "$release/INSTALL-SHA256SUMS" || \
        fail "version collision: bump lab-control VERSION"
    rm -rf "$stage"
else
    mv "$stage" "$release"
fi
trap - EXIT HUP INT TERM

temporary_link=$base/.current-$$
command_link=/usr/local/bin/.lab-control.$$
[ ! -e "$temporary_link" ] && [ ! -L "$temporary_link" ] || \
    fail "temporary current link already exists"
[ ! -e "$command_link" ] && [ ! -L "$command_link" ] || \
    fail "temporary command link already exists"
if [ -e "$base/current" ] || [ -L "$base/current" ]; then
    [ -L "$base/current" ] || fail "current path is not a symbolic link"
fi
[ ! -e /usr/local/bin/lab-control ] || [ -L /usr/local/bin/lab-control ] || \
    fail "/usr/local/bin/lab-control is not a symbolic link"
trap 'rm -f "$temporary_link" "$command_link"' EXIT HUP INT TERM
ln -s "releases/$version" "$temporary_link"
ln -s /opt/hil/control/current/ci/hil/lab_control.py "$command_link"
mv -Tf "$temporary_link" "$base/current"
mv -Tf "$command_link" /usr/local/bin/lab-control
trap - EXIT HUP INT TERM
(cd "$release" && sha256sum -c INSTALL-SHA256SUMS >/dev/null)
printf 'controller primitives version=%s release=%s\n' "$version" "$release"
