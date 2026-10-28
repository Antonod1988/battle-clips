# FLOW-1 «Battle Clips: pick two fighters and a stage, get the fight» (Claude Opus lane, owner 28.10; private repo)

Repo D:/pythonProject4/battle-clips, branch main (GitHub Antonod1988/battle-clips, PRIVATE). Report .planning/reports/FLOW-1.md.
Result line DRAFT READY <sha> <mp4 paths> / STOP <reason>. Commit on main with explicit paths as you go; never push (the lead pushes).

Owner's words:
- 01:10 «Нам нужно будет сделать из этого эксперимента универсальный флоу, чтобы проще потом было похожие делать».
- now, after version 4: «неплохо, но сильно колокольные звуки оружия, и можно активнее, масштабнее, когда сила героя позволяет. Сделай
  гитхаб проект, как мы это сделали на примере Скорпиона и Мортала, чтобы могли потом проще такие же создавать, только выбирать
  любого бойца».

## Read first (read-only, copy code into this repo; never commit in those worktrees)
- D:/pythonProject4/crazy-fantasy-arena-mk4 (branch claude/versus-mk4, 3c155857): clips/versus4/*, clips/audio/*,
  .planning/reports/VERSUS-MK4.md. Version 4, the owner's latest: Scorpion vs Sub-Zero, 114 s, weapons, magic, temple stage.
- D:/pythonProject4/crazy-fantasy-arena-hulk (branch claude/versus-hulk, 9887e1f2): clips/hulk/*, .planning/reports/VERSUS-HULK.md.
  Hulk vs Captain America, 97 s, a city block with four areas.
- clips/versus3 and .planning/reports/VERSUS-1.md in either worktree: version 3 (the look the owner approved) and the audio module.
- Output folders with frames and sheets: D:/pythonProject4/lab-review/«2026-10-28 versus-v4», «2026-10-28 hulk-vs-cap».
- The SOUND-2 lane is fixing the weapon sounds in clips/audio/sfx.py on claude/versus-mk4 right now. Take clips/audio from there
  when the lead tells you it landed (before you mix version 5).

## What to build
1. **One engine, many fighters.** The three clips are three diverging copies of the same modules (px, rig, body, moves, fx, magic,
   stage, scene, script, render, text). Merge them into one shared engine. A clip is then a recipe, not a code folder.
2. **A fighter is a data file.** fighters/<name>.py (or yaml): look (body build and size class, palette, mask, hood, helmet, hair,
   cape, emblem), weapons (drawn mid-fight, thrown, lost, caught), powers by element (fire, ice, lightning, raw strength and
   shockwaves, shield ricochet, rope or grapple, teleport, energy...), signature moves, voice lines, a finisher, and a **power tier**
   (1 trained human like Captain America, 3 magic warriors like the MK ninjas, 5 titans like the Hulk). Engine code changes only when a
   fighter needs a truly new kind of move; then the new move goes into the shared library for everyone.
   Fan fighters (Scorpion, Sub-Zero, Hulk, Captain America) live in fighters/fan/. Add fighters/_template with comments.
3. **A stage is a data file too.** stages/: the temple (v4), the city block (Hulk), the courtyard (v3). Areas on different levels,
   routes between them (leap, fall through a floor, run over a bridge, smash through a wall), breakables, hazards, ambience.
4. **A recipe per clip.** recipes/<clip>: title, the two fighters, the stage, length, seed, winner, finisher, music mood, and optional
   pinned beats (the moments that must happen, like «GET OVER HERE!»). Everything not pinned comes from the choreographer.
5. **The choreographer.** Builds the fight from both move sets, the stage routes and the power tiers: intro, feeling out, weapons,
   magic, travel to a new area, escalation, turning point, climax, finisher, win card. Rules the owner gave across versions: no idle
   stomping (in guard they circle, stalk, reposition), something lands, clashes or travels every 2-3 seconds, cartoon motion
   (anticipation, squash and stretch, smears, arcs, bounces), blood and damage that build up and stay, at least three areas in a
   two-minute clip, no move repeated back to back. Same seed, same fight; another seed, another fight.
6. **Power scale** («масштабнее, когда сила героя позволяет»): the tier drives leap height and length, knockback (through walls, across
   the area), magic size (a wall of hellfire across the courtyard, the whole floor frozen, ice spikes as tall as the temple, a fire
   tornado), destruction size (crack, hole, a building part collapsing), shake and slow motion. Mixed tiers fight differently: the
   weaker one dodges, uses weapons and the stage; the stronger one smashes.
7. **One command.** `python make.py recipes/<clip>` renders, makes the sound, encodes the 1080p mp4, a copy under 30 MB and the
   moments sheet, into out/<clip>/ and a lab-review folder. Add `--preview` (fast, low quality, no sound) and `--from/--to` seconds.
8. **The v4 limits, if they fit:** fighters face only left or right (add a front and a back three-quarter facing at least for moving
   and striking); falling pillar and bridge pieces slide flat like stairs (rotate them as bodies); ice drawn as plain shapes (facets and
   highlights). Say in the report which ones you did.
9. **Docs for the owner:** README.md in Russian, plain words, short: what the project is, how to make a clip with one command, how
   to add a fighter, a stage, a recipe, where the files land, why the repo stays private (the fan fighters). Code comments in English.
   requirements.txt and the ffmpeg path setting in one place.

## Acceptance
- **Version 5**, recipes/scorpion-vs-subzero: 110-125 s, made only with make.py. Keep what the owner liked in v4 (swords, the ice
  sword and hammer, teleport, hellfire, the ice clone, the roof, the bridge, the kunai through the shoulder with «GET OVER HERE!»,
  «FINISH HIM», the fire fatality, «FATALITY», «SCORPION WINS»); more hits than v4 (48 + 17 clashes); magic and destruction visibly
  bigger; the fixed weapon sounds. Files into D:/pythonProject4/lab-review/2026-10-28 versus-v5/.
- **Recipe-only test**: Sub-Zero vs Captain America on the city stage, about 60 s, made by writing only a recipe file, no new engine
  code. Files into D:/pythonProject4/lab-review/2026-10-28 subzero-vs-cap/. In the report: what was missing or awkward.
- The next lane (FLOW-2) will add a brand-new fighter (Spider-Man) with only the docs to guide it, so the docs must be enough.

MACHINE: no game window; Python and ffmpeg -threads 2; one render at a time. PROCESS RULE: never stop a process you did not start.
Never put mp4, m4a or wav into git. Report in English: the result line, the repo layout, how long a clip takes from recipe to files,
version 5 by time, the recipe-only test and its gaps, what is left.
