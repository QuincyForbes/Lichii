#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# One-time check of the Mac and the phone for the iOS bench; prints exactly what is missing.
#   bash setup-mac.sh [--bundle-id <id>] [--team-id <team>] [--device <udid>]
# Writes ~/lichii-bench/config.json (device, bundle_id, team_id) and runs bench_mac.py doctor.
# Never asks for or stores a password: signing uses the Apple ID already signed into Xcode.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bundle= team= device=
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle-id) bundle=$2; shift 2 ;;
    --team-id) team=$2; shift 2 ;;
    --device) device=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
missing=0
ok() { echo "  ok    $1"; }
miss() { echo "  MISSING $1"; echo "          -> $2"; missing=$((missing + 1)); }

echo "Mac"
[ "$(uname)" = Darwin ] || { echo "not a Mac"; exit 1; }
ok "macOS $(sw_vers -productVersion)"
if dev=$(xcode-select -p 2>/dev/null) && [ -d "$dev" ]; then
  if [[ "$dev" == *CommandLineTools* ]]; then
    miss "Xcode selected (only the Command Line Tools are)" \
         "install Xcode 26 from the App Store, then: sudo xcode-select -s /Applications/Xcode.app"
  else
    version=$(xcodebuild -version 2>/dev/null | head -1)
    major=$(echo "$version" | sed -E 's/Xcode ([0-9]+).*/\1/')
    if [ "${major:-0}" -ge 26 ]; then ok "$version ($dev)"; else
      miss "Xcode 26 or later (found ${version:-none})" "an iOS 26 phone needs Xcode 26's device support"; fi
  fi
else
  miss "Xcode" "install Xcode 26 from the App Store and open it once (it installs its components)"
fi
if xcodebuild -checkFirstLaunchStatus >/dev/null 2>&1; then ok "Xcode first-launch components"; else
  miss "Xcode first-launch components" "sudo xcodebuild -runFirstLaunch"; fi
for tool in devicectl xctrace lldb codesign; do
  if xcrun --find "$tool" >/dev/null 2>&1; then ok "xcrun $tool"; else
    miss "xcrun $tool" "comes with Xcode; select Xcode with xcode-select -s"; fi
done
if command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
  ok "python3 $(python3 -V 2>&1 | cut -d' ' -f2)"
else
  miss "python3 3.9+" "xcode-select --install"
fi
identities=$(security find-identity -v -p codesigning 2>/dev/null | grep -E '"Apple Development' || true)
if [ -n "$identities" ]; then ok "signing identity: $(echo "$identities" | head -1 | sed -E 's/.*"(.*)"/\1/')"; else
  miss "an Apple Development certificate in the keychain" \
       "Xcode > Settings > Accounts: sign in with the Apple ID, Manage Certificates > + > Apple Development"
fi
if [ -z "$team" ] && [ -n "$identities" ]; then
  cert_name=$(echo "$identities" | head -1 | sed -E 's/.*"(.*)"/\1/')
  team=$(security find-certificate -c "$cert_name" -p 2>/dev/null | openssl x509 -noout -subject 2>/dev/null |
         sed -nE 's/.*OU ?= ?([A-Z0-9]{10}).*/\1/p')
fi
if [ -n "$team" ]; then ok "team $team"; else
  miss "the development team id" "pass --team-id <10 characters> (developer.apple.com > Membership)"; fi
if [ -z "$bundle" ] && [ -f ~/lichii-bench/config.json ]; then
  bundle=$(python3 -c "import json;print(json.load(open('$HOME/lichii-bench/config.json')).get('bundle_id',''))")
fi
if [ -z "$bundle" ] && [ -n "$team" ]; then
  bundle="dev.lichii.bench.$(echo "$team" | tr 'A-Z' 'a-z')"
fi
echo "  info  bench bundle id $bundle (a separate app with its own Documents: the owner's Lichii"
echo "        install, settings and saves are not touched)"

mkdir -p ~/lichii-bench
python3 - "$HOME/lichii-bench/config.json" "$bundle" "$team" "$device" <<'EOF'
import json, os, sys
path, bundle, team, device = sys.argv[1:5]
config = json.load(open(path)) if os.path.exists(path) else {}
for key, value in (("bundle_id", bundle), ("team_id", team), ("device", device)):
    if value:
        config[key] = value
json.dump(config, open(path, "w"), indent=2)
print("  info  wrote " + path + ": " + json.dumps(config))
EOF

echo
echo "Phone"
python3 -I "$here/bench_mac.py" doctor
doctor=$?
echo
if [ $missing -gt 0 ] || [ $doctor -ne 0 ]; then
  echo "Fix the MISSING lines above and run setup-mac.sh again. Profile: bench_mac.py make-profile."
  exit 1
fi
echo "Ready. Check that bench mode starts: python3 -I $here/bench_mac.py probe"
