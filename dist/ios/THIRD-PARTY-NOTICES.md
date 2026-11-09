# iOS app: third-party notices

Parts of the iOS app (`src/ios_frontend`) are adapted from these projects. Each adapted file names
its source in its header. One line per adapted file or component.

## MeloNX

https://git.ryujinx.app/projects/MeloNX @ 6a1c159, Copyright Stossy11 and MeloNX contributors.
License: MIT (`LICENSES/MeloNX-MIT.txt`); the project README states GPLv3.

- `App/Content/OnboardingView.swift`: setup pages, ContinueButton and the backslide transition
  (`UI/OnBoarding/SetupView.swift`, `Pages/KeysView.swift`, `Pages/FirmwareView.swift`).
- `App/Content/GuestDialogs.swift`: guest dialogs as UIAlertControllers over the top view
  controller (`Common/AlertHandlers.swift`).

## Melo-Controller

https://github.com/stossy11/Melo-Controller, Copyright Stossy11.
License: GPLv3 (`LICENSES/Melo-Controller-GPLv3.txt`).
