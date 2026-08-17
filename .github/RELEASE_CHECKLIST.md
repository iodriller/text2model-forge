# Release checklist

Release version: `____________`  Commit: `________________________________________`

- [ ] Version and signed tag match (`0.2.0rc1` → `v0.2.0-rc.1`).
- [ ] `CHANGELOG.md` moves the release contents out of `Unreleased` and records the release date.
- [ ] Working tree is clean and the release commit is on the protected default branch.
- [ ] Windows, Ubuntu, and macOS CI checks pass from clean environments.
- [ ] Full randomized tests and deterministic D0–D10 demo pass.
- [ ] Wheel installs outside the checkout and loads all 13 worker manifests.
- [ ] Compose validates; the pinned non-root image builds and becomes healthy.
- [ ] Dependency audit, CodeQL, and all-history Gitleaks checks pass.
- [ ] Every dependency/model in the selected live stack has an exact revision and reviewed terms.
- [ ] All ten `static-props-8gb-v1` cases were attempted with real workers.
- [ ] `text2model-forge golden evaluate` reports at least 8/10 passing cases.
- [ ] The generated evidence gallery and machine/model metadata are attached.
- [ ] The owner accepted public exposure of historical author names/emails.
- [ ] The owner completed any required trademark/name and legal review.
- [ ] Release artifacts contain wheel, sdist, source archives, checksums, two SPDX SBOMs, and attestations.
- [ ] Security reporting, branch protection, issue forms, and Discussions/issue policy are configured.
- [ ] README claims match the evidence attached to this exact commit.
- [ ] Repository visibility changes only after every item above is checked.
