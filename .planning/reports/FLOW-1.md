# FLOW-1 «Battle Clips: pick two fighters and a stage, get the fight»

Status: DRAFT READY (result line at the end). Repo D:/pythonProject4/battle-clips, branch main, committed with explicit
paths, never pushed. Fan fighters: the repository and the clips stay private.

## Repo layout

```
make.py                 one command: recipe -> mp4 with sound, copy under 30 MB, sheets (--preview, --from/--to,
                        --stills, --plan)
config.py               ffmpeg path, threads (2), out/ and the lab-review folder, in one place
requirements.txt        numpy, scipy, pillow
README.md               for the owner, in Russian
engine/                 the one shared engine
  px.py rig.py text.py proj.py   pixels, the skeleton (with a build's shoulder/hip spread), words, the projection
  body.py               the costume painter, all data-driven: materials, limb bands, torso, emblems, four head kinds
                        (ninja, helmet, full mask with lenses and web lines, bare with hair), tabards, a cape,
                        scabbards, rope, a round shield; wounds, bruises, frozen, burnt, skull, skeleton, x-ray
  moves.py              every move of the three clips merged (renamed where they clashed), stances (ninja, shield,
                        brute), the calm guard (no hop on the spot)
  fx.py magic.py        particles and set pieces of both clips; magic with sizes as parameters, plus new big forms
                        (giant ice spikes, the whole floor freezing, a fire tornado, a wall of hellfire)
  scene.py              World and Fighter for any two fighters on any stage (per-fighter size, stance, tier,
                        three-quarter turning)
  acts.py               primitives: strike (knockback and effects scaled by tier, kept inside the area), clash,
                        whiff, leaps, falls, burning, craters
  beats.py              the beat library (38 beats: exchanges, blades, powers by element, travel, climaxes,
                        finishers)
  choreo.py             the choreographer
  render.py sound.py    frames to files; the sound (look-ahead true-peak limiter, then -14 LUFS mastering)
  fighter.py            loads and checks fighter files
fighters/_template.py   every option, commented
fighters/fan/           scorpion.py subzero.py captain_america.py hulk.py
stages/                 temple.py (version 4), city.py + city_props.py (the Hulk clip), courtyard.py (version 3;
                        its comment documents the stage protocol)
recipes/                scorpion-vs-subzero.py (version 5), subzero-vs-cap.py (the recipe-only test), _template.py
audio/                  the shared audio module with SOUND-2's sfx.py (652f2f9f); sfx_city.py (the Hulk clip's kinds,
                        the shield made dry like SOUND-2's steel), scores.py (moods: ninja, heroic)
tools/                  count.py (blows, length, standing still, off-surface, per beat, without drawing),
                        kit.py (lists everything a fighter or recipe can name; draws a fighter in three views;
                        shows a stage), sound_check.py (SOUND-2's clash regression check, kept)
```

## How the flow works

- A fighter is a data file: look, tier (1 trained human, 3 magic warrior, 5 titan), size, stance, strikes by kind,
  weapons (drawn or formed, a heavier one after the first breaks), powers (beats by element), signature, breakout,
  finisher, entrance, voice.
- A stage is a module: the picture plus AREAS (bounds, floor height, centre), ROUTES (leap, knocked_off,
  thrown_through, run_across, smash_through), targets() (what can be smashed, by kind), set pieces.
- A recipe: fighters, stage, length, seed, winner, finisher, music, pinned beats. Everything not pinned comes from the
  choreographer.
- The choreographer runs acts (intro, opening, travel, weapons, magic, travel, escalation, turning point, climax,
  finish), each with a share of the length in video seconds (slow motion counted, stepping exactly as the renderer).
  Pins happen in their act; when a recipe pins nothing for an act, it pins each kit's own specials itself (fewer in
  short clips). The rest is filled with beats from both kits by weight: never the same beat twice in a row, a special
  shows once, after every special comes close fighting, the winner leads a little more often but nobody leads four
  beats running, the loser takes the turning point. The fight is planned again with the same seed, its generated part
  shortened or lengthened, until the video length is within 2 s of the recipe's. Same seed, same fight.
- Rules from the owner built in: the guard only rolls the weight, idle stretches over 0.45 s become small stalks;
  strings, clashes or travel every couple of seconds; cartoon motion and smears from version 3/4; blood and wounds
  stay; several areas per clip.
- Power scale: tier drives knockback (a titan barely moves when a man hits him, a man flies when a titan does),
  leap height, the size of hellfire columns, ice walls, spikes, the tornado, craters, shake. Mixed tiers: the weaker
  one gets shield braces, acrobatic dodges, ricochets off the scenery; the stronger one gets the heavy smashes.

## Time from recipe to files (this laptop, ffmpeg -threads 2, one render at a time)

- `--plan`: 5 s (the stage image is built once, then planned 2-5 times to fit the length).
- `tools/count.py` (the whole fight stepped without drawing): about 10 s.
- `--preview` of the 2-minute clip (15 fps, 960x540, no sound): about 1 min.
- Full make of version 5 (121.5 s): frames 2 min 29 s, sound about 1 min 15 s, the two-pass copy about 3 min, sheets
  and moments: 7 min 35 s, plus 1 min 17 s for the sound once the true-peak limiter was added (remade with
  `python engine/sound.py out/scorpion-vs-subzero versus-v5 ninja`; a fresh make now does it in one go, about 8.5 min).
- Full make of the 60 s test: 3 min 20 s (frames 67 s).

## Version 5 (recipes/scorpion-vs-subzero.py, made only with make.py)

D:/pythonProject4/lab-review/2026-10-28 versus-v5/: versus-v5-1080p.mp4 (1920x1080, 30 fps, 3645 frames, 121.5 s, H.264
CRF 17 + AAC, 34.0 MB), versus-v5-under-30mb.mp4 (28.9 MB, two-pass 1645 kbps + the same track), versus-v5-moments.png,
versus-v5-contact.png, moments/ (each moment at 1080p), versus-v5-track.m4a (-14.0 LUFS, true peak -1.2 dBFS),
versus-v5-cues.json, versus-v5-log.txt (the beats the choreographer chose).

Blows counted as they play: 60 land (8 of them blade cuts), 23 blade clashes, 5 blocked. Version 4: 48 + 17 + 4.
Seed 20 was picked with tools/count.py among seeds at the same length (others give 50-66 blows, 15-28 clashes).

| Video s | What happens |
| --- | --- |
| 0-4.7 | Title over the pan from the pinnacle to the courtyard; Scorpion out of fire, Sub-Zero flips down from up high in frost; FIGHT! |
| 5.0-7.6 | Teleport punch (pinned): Scorpion folds into fire, drops out of it behind Sub-Zero. |
| 7.6-21.1 | Opening: a two-sided exchange, flying kicks meeting in mid-air (twice), an ice ball flipped over, Sub-Zero knocked into a brazier (it topples, coals burn to the end), a sweep jumped and punished. |
| 21.1-30.6 | Through the balustrade down to the ledge over the drop; the freezing ground wave jumped, an axe kick and a string; an 8-blow exchange; a leap back up. |
| 30.6-33.4 | Exchange in the courtyard. |
| 33.4-38.4 | Swords drawn from the back (glints), the ice sword formed; five clashes straight away. |
| 38.4-41.9 | Blades locked, hop apart, the clash as they pass, the cut opens on Sub-Zero's arm. |
| 41.9-45.1 | The disarm: a sword spins away and sticks in the floor; the other blade shatters the ice sword; the ice hammer formed. |
| 45.1-49.5 | The hammer craters the floor (bigger than version 4), drives Scorpion back with two parried swings; the pillar takes the third and falls, turning as a body. |
| 49.2-52.7 | The sword kicked up and caught; an exchange. |
| 52.7-57.0 | The kunai batted away by the hammer; six clashes. |
| 57.0-61.9 | Hellfire: six columns about 120 px tall (version 4: 74) with two fans of columns across the courtyard, stopped by a taller ice wall in steam; the fire dash through the wall. |
| 61.9-65.5 | New: the floor freezes outward and a forest of giant faceted ice spikes bursts under Scorpion and throws him high. |
| 65.5-69.2 | New: a fire tornado tall as the hall lifts Sub-Zero, burns him three times and throws him. |
| 69.2-75.6 | Up on the temple roof; the hammer blows a hole in it; down the slope and off the eave. |
| 75.6-81.4 | Scorpion's cut combo (thrust, X cut), a counter. |
| 81.4-88.5 | The ice clone: Scorpion's cut freezes him solid, two hammer blows on the ice, the eyes burn, the fire breakout, the fury string. |
| 88.5-91.3 | Sub-Zero runs for the bridge and leaps the gap; Scorpion follows in a fire dash. |
| 91.3-95.6 | On the pinnacle: four clashes, the X cut into the gong (it rings). |
| 95.6-99.3 | The hammer uppercut throws Scorpion onto the bridge, the planks smashed, the bridge snaps; both halves turn down as bodies; both leap for their own side. |
| 99.3-106.6 | The kunai across the chasm through the shoulder, «GET OVER HERE!», the yank over the void, two blows, the slow-motion uppercut, the air punch, the axe kick, the slam (crater). |
| 106.7-121.5 | FINISH HIM, the burning skull, the fire breath, FATALITY, the charred skeleton collapsing into bones, SCORPION WINS. |

Kept from version 4: swords, the ice sword and hammer, teleport, hellfire against the ice wall, the ice clone, the
roof, the bridge, the kunai through the shoulder with «GET OVER HERE!», FINISH HIM, the fire fatality, FATALITY,
SCORPION WINS. Weapon sounds: SOUND-2's dry steel (taken from claude/versus-mk4 652f2f9f); the shield's ring dried the
same way. Not listened to: measured only.

## The version 4 limits

- Three-quarter facing: done for moving and striking. When an action travels mostly in depth the body turns towards
  the camera or away (smoothly): the side view shortens, the face slides round (both eyes, the helmet's letter, the
  emblem move to the front), from behind the head is the hood or hair and the scabbards, the shield and a cape are
  drawn over the body. It is a turn of the side-view rig, not a separate front pose.
- Falling pillar and bridge pieces: done. The pillar is drawn through its real 3D axis as it falls sideways and towards
  the camera (it foreshortens, keeps its width, the lit side stays up-left); the snapped bridge halves turn down about
  their anchors with every plank and both ropes re-projected in depth, not a flat ladder.
- Ice as facets: done for the ice sword (split planes, a bright ridge, inner cracks), the hammer (three faces by
  normal, edges, cracks) and the new giant spikes; the ice wall, block and clone are still version 4's shapes.

## The recipe-only test (recipes/subzero-vs-cap.py, nothing pinned)

D:/pythonProject4/lab-review/2026-10-28 subzero-vs-cap/: subzero-vs-cap-1080p.mp4 (1920x1080, 59.4 s, 1781 frames,
18.3 MB), subzero-vs-cap-under-30mb.mp4 (15.7 MB), sheets, moments/, the track (-14.0 LUFS, -1.5 dBTP), the cue
sheet, the log. 29 blows land, 1 clash, 6 taken on the shield; no stretch of standing still, nobody off a surface.

Video seconds: 0-4.8 title over MAIN ST, Captain America walks in, Sub-Zero flips down, FIGHT!; 4.9-6.8 flying kicks
meet; 6.8-9.6 Cap's string knocks Sub-Zero onto a car (dented, glass, alarm); 9.6-11.9 the ground wave; 11.9-15.0 Cap
thrown through the diner's west wall (the hole stays); 15.0-21.2 the ice hammer formed, the floor cratered, two hammer
blows taken on the shield, a column cracked by the third; 21.2-24.1 the diner floor freezes, Cap slips and is kicked
along the ice; 24.1-25.5 Cap's shield rings off the diner's furniture into Sub-Zero's head and back to his hand;
25.5-29.2 an exchange; 29.2-32.4 Cap thrown through the east wall onto 5TH AVE; 32.4-40.9 the turning point: Cap has
him down with a string and a counter, Sub-Zero rises with fury; 40.9-46.9 the ice spikes throw Cap high, the uppercut
and the slam; 47-59.4 FINISH HIM, Cap frozen solid and shattered with the ice, FATALITY, SUB-ZERO WINS.

Missing or awkward when writing only the recipe (what was fixed in the engine during this lane is marked):
- Before auto pins the choreographer filled an unpinned fight with generic hand-to-hand only; it now pins each kit's
  specials itself (engine, fixed).
- Captain America's shield moves existed only as hand-written script in the Hulk clip; shield_throw, shield_brace,
  acrobat_dodge and shield_combo had to become beats (engine, fixed). The Hulk's grabs (ankle swing, throat carry)
  and car lifting and throwing are still not beats.
- The finisher comes from the winner's file, so a Marvel fight ends with MK's FINISH HIM and FATALITY; the knockout
  ending has no card of its own besides the win card.
- The city's subway is drawn but no route reaches it (no cave-in beat); the diner's slanted walls make its area
  bounds approximate; the title pan is one per stage.
- A minute is short: a third of it goes to the title and the fatality, so the blow rate is lower than version 5's.
- The music mood has to be named in the recipe (heroic for Marvel); voices exist only for the announcer, Scorpion and
  the Hulk.

## What is left

- Not listened to: both tracks are measured (loudness, true peak, no unknown cue kinds), the owner's ears are the check.
- The Hulk is a data file with his strength beats (shockwave, leap smash, roar, the brute stance); he was only
  checked with the planner and tools/count.py (Hulk against Captain America on the courtyard and the city), not
  rendered. His grabs need the Hulk clip's attachment code merged into the scene.
- No Spider-Man yet (FLOW-2): the template already offers the full mask with lenses and web lines, the spider emblem
  and a cape (checked with tools/kit.py); a web line that pulls or swings needs a new beat (fx.Spear is a rope with a
  blade; a web variant is the natural start).
- Three-quarter facing is a turned side view; a true front pose would need front-facing moves.
- The ice wall, the freeze block and the clone are still plain shapes.
- Blow counts depend on the seed (50-66 at 122 s for version 5); pick seeds with tools/count.py.
- Nothing pushed (the lead pushes).

Result: DRAFT READY (the commit that adds this report on main)
D:/pythonProject4/lab-review/2026-10-28 versus-v5/versus-v5-1080p.mp4
D:/pythonProject4/lab-review/2026-10-28 versus-v5/versus-v5-under-30mb.mp4
D:/pythonProject4/lab-review/2026-10-28 subzero-vs-cap/subzero-vs-cap-1080p.mp4
D:/pythonProject4/lab-review/2026-10-28 subzero-vs-cap/subzero-vs-cap-under-30mb.mp4

## HULK-2: Hulk vs Captain America, version 2

The owner wanted version 1 improved, not a Sub-Zero test («халка бы улучшил просто видео»). Remade through
battle-clips as recipes/hulk-vs-cap.py: the city, 116 s, seed 2, winner the Hulk, universe 'marvel', finisher
titan_smash, the heroic score. 28 beats, 26 of them pinned in version 1's order through all four areas.

Media, D:/pythonProject4/lab-review/2026-10-28 hulk-vs-cap-v2/:
- hulk-vs-cap-v2-1080p.mp4: 1920x1080, 30 fps, 116.0 s, 36.5 MB; hulk-vs-cap-v2-under-30mb.mp4: 27.8 MB.
- hulk-vs-cap-v2-moments.png (56 moments) and moments/; the contact sheet, the track, the cue sheet, the log.

Blows: 75 land and 2 are taken on the shield (version 1: 55 and 3). Standing still: two stretches of about half a
second (Cap 40.5, the Hulk 52.9); nobody stands on nothing. Sound: -14.0 LUFS, true peak -1.2 dBFS, every cue kind
registered; the city effects are made dry like SOUND-2 (short rooms, metal that no longer rings for a second);
«FIGHT!», «HULK SMASH!» and «HULK WINS» in English.

| Video s | What happens |
| --- | --- |
| 0-5.4 | MAIN ST. The Hulk drops out of the sky (0.9): crater, cars hop, windows burst; the title; FIGHT! |
| 6.1-13.9 | Cap's opening string (9 blows) and a juggle (5). |
| 13.9-19.5 | The backhand onto the blue car (15.9); the charge, Cap's slide, the Hulk trips into a lamp post (18.5). |
| 19.5-21.9 | The shield off a lamp and a shop window into the Hulk's head and back. |
| 21.9-25.4 | The Hulk lifts and throws the blue car; Cap flips over it; it goes through the pharmacy and its front collapses (24.9). |
| 25.4-34.7 | Cap's counter; the Hulk catches his kick, slams him both sides (30.4) and throws him through the diner's west wall (32.8); charges through after him (33.9). |
| 34.7-53.5 | THE DINER: the table kick (35.9), the shield cuts a pendant down on his head (37.8), the blows on the shield, the ground smash (41.5), Cap into the column (43.7), an 8-blow exchange, the throat, headbutts and the stomp (51-52.7). |
| 53.5-57.8 | The thunderclap blows Cap through the east wall (55.7); the Hulk walks through (57.0). |
| 57.8-72.6 | 5TH AVE: the leap crushes the silver car and shakes the BANK sign loose (59.3); Cap off a car's roof (62.8); the shield off a window and a car; the slab on the shield (68.9); Cap swatted through the bank's front, which collapses (71.6). |
| 72.6-76.9 | The Hulk comes down from the sky; the road caves into the subway (74.7), the water main bursts. |
| 76.9-89.6 | THE SUBWAY: Cap's 10-blow string, the rage (81.3), the third rail (82.9), the shield brings a light down, the Hulk driven into a steel column (88.3). |
| 89.6-100.0 | By the throat up through the hole (92.3), slammed down (93.5); Cap's last stand: the shield off a car and a lamp, the leaping kick (98.7). |
| 100.0-116.0 | HULK SMASH! (101.7): the haymaker on the shield, the street blown apart, the hanging signs come down, Cap blasted into a car; he falls back (108.1), the roar, HULK WINS (109.5), the end card (113.3). |

Bigger than version 1: 19 s longer and 20 more blows; two building fronts collapse into ragged openings with rubble
that stays (the pharmacy under the car, the bank under Cap; version 1 had a hole and a crater); the leap lands on a
car and crushes it (version 1 flipped one beside him); hanging signs fall (under the leap and in the final blast);
Cap's strings are 9 and 10 blows (version 1: 8). The rest of version 1's set pieces (the cave-in and water main, the
two diner walls, the third rail, the leap back up) are kept, now as shared beats: ankle_slam, throat_slam, car_throw,
chunk_throw, ground_smash, rage, charge_trip for any tier-5 fighter; long_combo, car_spring, table_kick, last_stand and
the ricocheting shield for a trained human; titan_smash and the end card for a Marvel ending (no FINISH HIM, no
FATALITY). Not listened to here: the owner's ears are the check.

Result: DRAFT READY (the commit that adds this section on main)
D:/pythonProject4/lab-review/2026-10-28 hulk-vs-cap-v2/hulk-vs-cap-v2-1080p.mp4
D:/pythonProject4/lab-review/2026-10-28 hulk-vs-cap-v2/hulk-vs-cap-v2-under-30mb.mp4
