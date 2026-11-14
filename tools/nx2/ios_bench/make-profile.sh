#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Makes the development provisioning profile bench_mac.py signs Lichii with: builds an empty app
# with Lichii's bench bundle id and memory entitlements once, with Xcode's automatic signing
# (the Apple ID signed into Xcode registers the device and the App ID capabilities). Reads
# bundle_id and team_id from ~/lichii-bench/config.json; DEVICE=<udid> picks the phone.
set -euo pipefail
config=~/lichii-bench/config.json
read_config() { python3 -c "import json,sys;print(json.load(open('$config')).get('$1',''))"; }
bundle=$(read_config bundle_id)
team=$(read_config team_id)
udid=${DEVICE:-$(read_config device)}
if [ -z "$bundle" ] || [ -z "$team" ]; then
  echo "bundle_id and team_id are needed in $config (setup-mac.sh writes them)" >&2
  exit 1
fi
if [ -z "$udid" ]; then
  udid=$(python3 "$(dirname "$0")/bench_mac.py" devices | python3 -c \
    "import json,sys;d=[x for x in json.load(sys.stdin) if x['platform']=='iOS'];print(d[0]['udid'] if d else '')")
fi
work=~/lichii-bench/signer
rm -rf "$work"
mkdir -p "$work/Signer.xcodeproj/xcshareddata/xcschemes"
cat > "$work/main.c" <<'EOF'
int main(void) { return 0; }
EOF
cat > "$work/Signer.entitlements" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>com.apple.developer.kernel.increased-memory-limit</key>
	<true/>
	<key>com.apple.developer.kernel.increased-debugging-memory-limit</key>
	<true/>
	<key>com.apple.developer.kernel.extended-virtual-addressing</key>
	<true/>
</dict>
</plist>
EOF
cat > "$work/Signer.xcodeproj/project.pbxproj" <<EOF
// !\$*UTF8*\$!
{
	archiveVersion = 1;
	classes = {
	};
	objectVersion = 56;
	objects = {
		A1000000000000000000000A = {isa = PBXBuildFile; fileRef = A1000000000000000000000B; };
		A1000000000000000000000B = {isa = PBXFileReference; lastKnownFileType = sourcecode.c.c; path = main.c; sourceTree = "<group>"; };
		A1000000000000000000000C = {isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = Signer.app; sourceTree = BUILT_PRODUCTS_DIR; };
		A1000000000000000000000D = {isa = PBXFileReference; lastKnownFileType = text.plist.entitlements; path = Signer.entitlements; sourceTree = "<group>"; };
		A10000000000000000000010 = {isa = PBXGroup; children = (A1000000000000000000000B, A1000000000000000000000D, A10000000000000000000011, ); sourceTree = "<group>"; };
		A10000000000000000000011 = {isa = PBXGroup; children = (A1000000000000000000000C, ); name = Products; sourceTree = "<group>"; };
		A10000000000000000000020 = {isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = (A1000000000000000000000A, ); runOnlyForDeploymentPostprocessing = 0; };
		A10000000000000000000030 = {isa = PBXNativeTarget; buildConfigurationList = A10000000000000000000041; buildPhases = (A10000000000000000000020, ); buildRules = (); dependencies = (); name = Signer; productName = Signer; productReference = A1000000000000000000000C; productType = "com.apple.product-type.application"; };
		A10000000000000000000031 = {isa = PBXProject; attributes = {LastUpgradeCheck = 1600; TargetAttributes = {A10000000000000000000030 = {CreatedOnToolsVersion = 16.0; }; }; }; buildConfigurationList = A10000000000000000000040; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base, ); mainGroup = A10000000000000000000010; productRefGroup = A10000000000000000000011; projectDirPath = ""; projectRoot = ""; targets = (A10000000000000000000030, ); };
		A10000000000000000000040 = {isa = XCConfigurationList; buildConfigurations = (A10000000000000000000050, ); defaultConfigurationIsVisible = 0; defaultConfigurationName = Debug; };
		A10000000000000000000041 = {isa = XCConfigurationList; buildConfigurations = (A10000000000000000000051, ); defaultConfigurationIsVisible = 0; defaultConfigurationName = Debug; };
		A10000000000000000000050 = {isa = XCBuildConfiguration; buildSettings = {SDKROOT = iphoneos; IPHONEOS_DEPLOYMENT_TARGET = 17.0; }; name = Debug; };
		A10000000000000000000051 = {isa = XCBuildConfiguration; buildSettings = {CODE_SIGN_STYLE = Automatic; CODE_SIGN_ENTITLEMENTS = Signer.entitlements; DEVELOPMENT_TEAM = $team; GENERATE_INFOPLIST_FILE = YES; INFOPLIST_KEY_UILaunchScreen_Generation = YES; PRODUCT_BUNDLE_IDENTIFIER = "$bundle"; PRODUCT_NAME = Signer; TARGETED_DEVICE_FAMILY = "1,2"; }; name = Debug; };
	};
	rootObject = A10000000000000000000031;
}
EOF
cat > "$work/Signer.xcodeproj/xcshareddata/xcschemes/Signer.xcscheme" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion = "1600" version = "1.7">
   <BuildAction parallelizeBuildables = "YES" buildImplicitDependencies = "YES">
      <BuildActionEntries>
         <BuildActionEntry buildForTesting = "NO" buildForRunning = "YES" buildForProfiling = "NO" buildForArchiving = "NO" buildForAnalyzing = "NO">
            <BuildableReference BuildableIdentifier = "primary" BlueprintIdentifier = "A10000000000000000000030" BuildableName = "Signer.app" BlueprintName = "Signer" ReferencedContainer = "container:Signer.xcodeproj">
            </BuildableReference>
         </BuildActionEntry>
      </BuildActionEntries>
   </BuildAction>
   <LaunchAction buildConfiguration = "Debug" selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB" launchStyle = "0" debugDocumentVersioning = "YES" allowLocationSimulation = "YES">
   </LaunchAction>
</Scheme>
EOF
cd "$work"
destination=()
if [ -n "$udid" ]; then destination=(-destination "id=$udid"); fi
xcodebuild -project Signer.xcodeproj -scheme Signer -configuration Debug -sdk iphoneos \
  "${destination[@]}" -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  -derivedDataPath "$work/dd" build | tail -25
profile=$(find "$work/dd" -name embedded.mobileprovision | head -1)
if [ -z "$profile" ]; then
  echo "no profile was made; see the xcodebuild output above, or make one in Xcode (README.md)" >&2
  exit 1
fi
mkdir -p ~/lichii-bench
cp "$profile" ~/lichii-bench/bench.mobileprovision
python3 - "$config" <<'EOF'
import json, os, sys
path = sys.argv[1]
config = json.load(open(path))
config["profile"] = os.path.expanduser("~/lichii-bench/bench.mobileprovision")
json.dump(config, open(path, "w"), indent=2)
EOF
security cms -D -i ~/lichii-bench/bench.mobileprovision | plutil -extract Entitlements xml1 -o - - | \
  grep -E "kernel|get-task-allow|application-identifier" -A1 | sed 's/^/  /'
echo "profile saved to ~/lichii-bench/bench.mobileprovision (config.json updated)"
