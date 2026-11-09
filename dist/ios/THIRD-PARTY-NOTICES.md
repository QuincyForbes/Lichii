# iOS app: third-party notices

Parts of the iOS app (`src/ios_frontend`) are adapted from these projects. Each adapted file names
its source in its header. One line per adapted file or component.

## MeloNX

https://git.ryujinx.app/projects/MeloNX @ 6a1c159, Copyright Stossy11 and MeloNX contributors.
License: MIT (`LICENSES/MeloNX-MIT.txt`); the project README states GPLv3.

- `Controls/PhysicalControllers.swift`: controller discovery, input binding and motion mapping
  (`Core/Controllers/ControllerManager.swift`, `NativeController.swift`, `Motion/BaseController+Motion.swift`)
- `Controls/RumbleController.swift`: CoreHaptics rumble (`Core/Controllers/Rumble/RumbleController.swift`)
- `Controls/ControllerSettingsView.swift`: per-player controller rows (`UI/Main/Settings/Elements/Controller/ControllerRow.swift`)
- `Controls/InGameMenuView.swift`, `Controls/ControlsOverlayView.swift`: in-game menu (`UI/Emulation/Config/InGameConfigView.swift`, `UI/Emulation/Metal/MetalViewContainer.swift`)
- `Controls/PerformanceHUD.swift`: performance overlay (`UI/Emulation/PerformanceDisplay/PerformanceOverlay.swift`, `Common/StatisticsHandler.swift`)
- `Controls/OrientationManager.swift`: rotation lock (`Common/OrientationManager.swift`)
- `Controls/ExternalDisplay.swift`: external display / AirPlay (`Common/AirPlay/Air.swift`)
- `Controls/TouchController.swift`: Controller protocol adapter (`Core/Controllers/VirtualController.swift`)

## Melo-Controller

https://github.com/stossy11/Melo-Controller, Copyright Stossy11.
License: GPLv3 (`LICENSES/Melo-Controller-GPLv3.txt`).

- `src/ios_frontend/Controls/MeloController/`: the whole package @ b16a8e3 (on-screen controller,
  layout editor and layout files), built as the `Melo_Controller` Swift module; Lichii's changes are
  marked "Lichii" in the files.
