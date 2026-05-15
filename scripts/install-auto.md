# Auto research on a 3-hour schedule (macOS launchd)

## One-time setup

1. **Seed the backlog** so the agent has something to research:
   ```
   mkdir -p .alpha_intern
   cp scripts/backlog.seed.txt .alpha_intern/backlog.txt
   ```
   Or add items by hand: `alpha-intern backlog-add "your hypothesis"`.

2. **Edit the plist.** Open `scripts/com.alpha-intern.auto.plist` and
   replace every `REPLACE_ME` with your username, then either:
   - Hardcode `ANTHROPIC_API_KEY` in the plist (simpler), or
   - Delete that key and instead run once:
     `launchctl setenv ANTHROPIC_API_KEY "sk-ant-..."`

3. **Install:**
   ```
   cp scripts/com.alpha-intern.auto.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.alpha-intern.auto.plist
   ```

4. **Verify it's loaded:**
   ```
   launchctl list | grep alpha-intern
   ```

## What it does

Every 3 hours, runs `alpha-intern auto --budget 1.0`. Each tick is:
- **Research turn (3 of every 4):** pops the top backlog item, runs the
  agent on it, writes a card to `.alpha_intern/cards/<run_id>.json`.
- **Meta-reflect turn (1 of every 4):** reads the last 12 cards plus
  existing memories/skills, asks the model for patterns, appends novel
  lessons/skills/hypotheses (deduped by title).

Daily $ cap: $1. If today's spend (sum of estimated USD across all
`llm_call` entries dated today UTC) exceeds the cap, the tick is
skipped — no API call is made.

## Inspecting

```
alpha-intern backlog-list        # what's still queued
alpha-intern usage --last 10     # token + $ cost per recent run
ls .alpha_intern/cards/          # per-run summaries
tail -f ~/Library/Logs/alpha-intern.auto.log
```

## Disable / uninstall

```
launchctl unload ~/Library/LaunchAgents/com.alpha-intern.auto.plist
rm ~/Library/LaunchAgents/com.alpha-intern.auto.plist
```
