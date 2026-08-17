#!/bin/sh
set -eu
set -f

fail() {
    printf 'target provision: %s\n' "$*" >&2
    exit 1
}

[ "$(id -u)" -eq 0 ] || fail "must run as root"
agent_source=${1:-}
authorized_key=${2:-}
execution_user=${3:-}
[ -f "$agent_source" ] && [ ! -L "$agent_source" ] || fail "unsafe agent source"
[ -f "$authorized_key" ] && [ ! -L "$authorized_key" ] || fail "unsafe public key"
case $execution_user in root|hilagent) ;; *) fail "execution user must be root or hilagent" ;; esac

install -d -m 0755 -o root -g root /etc/hil
identity_file=/etc/hil/target-execution-user
if [ -e "$identity_file" ] || [ -L "$identity_file" ]; then
    [ -f "$identity_file" ] && [ ! -L "$identity_file" ] || \
        fail "execution identity marker is unsafe"
    recorded_identity=$(sed -n '1p' "$identity_file")
    [ "$recorded_identity" = "$execution_user" ] || \
        fail "execution identity change requires explicit deprovision"
fi

awk 'NF { count++ } END { exit count == 1 ? 0 : 1 }' "$authorized_key" || \
    fail "public key file must contain exactly one non-empty line"
# Deliberate field splitting of one public-key line; globbing is disabled.
# shellcheck disable=SC2046
set -- $(sed -n '1p' "$authorized_key")
[ "$#" -ge 2 ] && [ "$1" = ssh-ed25519 ] || fail "only an ssh-ed25519 key is accepted"
case $2 in ''|*[!A-Za-z0-9+/=]*) fail "public key payload is malformed" ;; esac
key="$1 $2"
version=$(sh "$agent_source" --version)
case $version in ''|*[!0-9.]*) fail "invalid target-agent version" ;; esac

install -d -m 0755 -o root -g root /usr/local/libexec
versioned_agent=/usr/local/libexec/hil-target-agent-$version
if [ -e "$versioned_agent" ] || [ -L "$versioned_agent" ]; then
    [ -f "$versioned_agent" ] && [ ! -L "$versioned_agent" ] || \
        fail "versioned agent path is unsafe"
    cmp -s "$agent_source" "$versioned_agent" || \
        fail "version collision: bump target-agent version"
else
    install -m 0755 -o root -g root "$agent_source" "$versioned_agent"
fi
if [ "$execution_user" = hilagent ]; then
    if ! id hilagent >/dev/null 2>&1; then
        useradd --system --create-home --home-dir /var/lib/hil-target --shell /bin/sh hilagent
    fi
    account_home=/var/lib/hil-target
    [ "$(awk -F: '$1 == "hilagent" {print $6}' /etc/passwd)" = "$account_home" ] || \
        fail "existing hilagent account has an unexpected home"
    account_group=$(id -gn hilagent)
else
    account_home=/root
    account_group=$(id -gn root)
fi
install -d -m 0750 /opt/hil/runs
if find /opt/hil/runs -mindepth 1 -print -quit | grep -q .; then
    fail "/opt/hil/runs must be empty before changing execution identity"
fi
chown "$execution_user:$account_group" /opt/hil/runs
install -d -m 0700 -o "$execution_user" -g "$account_group" "$account_home/.ssh"

authorized_keys=$account_home/.ssh/authorized_keys
temporary=$account_home/.ssh/.authorized_keys.$$
trap 'rm -f "$temporary"' EXIT HUP INT TERM
umask 077
if [ -f "$authorized_keys" ] && [ ! -L "$authorized_keys" ]; then
    grep -v ' modelzoo-hil-target-agent$' "$authorized_keys" >"$temporary" || true
elif [ -e "$authorized_keys" ] || [ -L "$authorized_keys" ]; then
    fail "authorized_keys is unsafe"
else
    : >"$temporary"
fi
printf '%s %s\n' \
    'restrict,command="/usr/local/libexec/hil-target-agent --forced"' \
    "$key modelzoo-hil-target-agent" >>"$temporary"
chmod 0600 "$temporary"
chown "$execution_user:$account_group" "$temporary"
mv "$temporary" "$authorized_keys"
trap - EXIT HUP INT TERM

identity_temporary=/etc/hil/.target-execution-user.$$
trap 'rm -f "$identity_temporary"' EXIT HUP INT TERM
printf '%s\n' "$execution_user" >"$identity_temporary"
chmod 0644 "$identity_temporary"
chown root:root "$identity_temporary"
mv "$identity_temporary" "$identity_file"
trap - EXIT HUP INT TERM

temporary_link=/usr/local/libexec/.hil-target-agent.$$
[ ! -e "$temporary_link" ] && [ ! -L "$temporary_link" ] || \
    fail "temporary target-agent link already exists"
if [ -e /usr/local/libexec/hil-target-agent ] || \
    [ -L /usr/local/libexec/hil-target-agent ]; then
    [ -L /usr/local/libexec/hil-target-agent ] || \
        fail "stable target-agent path is not a symbolic link"
fi
trap 'rm -f "$temporary_link"' EXIT HUP INT TERM
ln -s "$(basename "$versioned_agent")" "$temporary_link"
mv -Tf "$temporary_link" /usr/local/libexec/hil-target-agent
trap - EXIT HUP INT TERM

printf 'target agent version=%s user=%s forced-command=enabled\n' \
    "$version" "$execution_user"
