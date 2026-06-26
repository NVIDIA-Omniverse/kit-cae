# Changelog

All notable changes to the CAE Extension Bundle will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.1.1]

- Updated the Kit-CAE release to support Kit SDK 110.1.2 as the default SDK, including refreshed app and
  streaming template support.
- Fixed bundle compatibility with Warp 1.13 by including the CAE data dtype/device helper updates required by
  the newer Warp API and the updated bundled DAV runtime.

## [2.1.0]

- Added dependency on `omni.cae.startup` so the bundled apps pick up the optional startup-USD-file behaviour.

## [2.0.0]

- Added dependency on `omni.cae.viz`.
- Added example scripts to the extension package along with testing for those example scripts.

## [1.0.0]

- Initial release of `omni.cae.bundle` extension
