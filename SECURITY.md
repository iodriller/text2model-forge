# Security policy

## Supported versions

Until the first stable release, only the newest tagged preview and the default
branch receive security fixes.

## Reporting a vulnerability

Do not open a public issue for a vulnerability, exposed credential, path
traversal, unsafe archive extraction, command injection, or network exposure.
Use GitHub's **Report a vulnerability** form in the repository Security tab.
Include the affected version, operating system, reproduction steps, impact, and
the smallest safe proof. Redact credentials, private prompts, generated assets,
and machine-specific paths.

The maintainer will acknowledge a report within seven days and will coordinate
validation, remediation, release, and credit through a private security
advisory. No bounty program is offered.

## Deployment boundary

Studio has no account or authorization system. It is designed for one trusted
operator and binds to loopback by default. Do not expose it to a LAN or the
public internet. `--allow-non-loopback` acknowledges that risk; use a separate
authenticated reverse proxy and network controls if you deliberately cross the
boundary.
