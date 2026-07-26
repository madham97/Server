# ngrok systemd units

Keeps both ngrok tunnels running as durable background services instead of
manually-started shell processes: `ngrok-http.service` (plain HTTP, no TLS
redirect — required by the Pi's GSM modem uploader) and `ngrok-https.service`
(default/https mode, for browser access to `/thermal`, `/config-help`, etc.).
Both run simultaneously against the same static reserved domain, split by
scheme — see `docs/decisions.md` for why two tunnels are needed.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/ngrok-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ngrok-http.service ngrok-https.service

# required, or both services die the next time you fully log out —
# without lingering, the user's systemd instance itself stops when there's
# no active login session
loginctl enable-linger "$USER"
```

Requires `ngrok` at `~/Server/ngrok` (repo root) and an authtoken already
configured (`ngrok config add-authtoken <token>` — see main README).

## Operate

```bash
systemctl --user status ngrok-http.service ngrok-https.service
systemctl --user restart ngrok-http.service   # or ngrok-https.service
journalctl --user -u ngrok-http.service -f    # or tail ~/ngrok.log / ~/ngrok2.log directly
```

## Why not just `nohup ... &`

That was the original approach and it's why this exists: processes started
that way are still tied to *some* parent shell/session lineage. One instance
happened to survive three weeks, but a later one, started via the CLI
harness's own background-task tracking, was killed outright when a session
boundary was crossed — silently, with no alert, breaking the Pi's uploads
until noticed. `systemd --user` + `loginctl enable-linger` detaches the
process from any particular session entirely and adds `Restart=always`, so a
crash (or the same kind of session-boundary kill) self-heals instead of
requiring a human to notice and restart it manually.
