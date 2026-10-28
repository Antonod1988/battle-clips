"""TEMPLATE for a clip. Copy to recipes/<clip>.py and run:  python make.py recipes/<clip>.py

Only 'fighters' and 'stage' are required; the choreographer builds the rest from the two fighters' kits and the
stage. Pin the moments that must happen; everything not pinned is generated (same seed, same fight).
"""

RECIPE = {
    'title': 'LEFT vs RIGHT',                 # for your notes (the title card uses the fighters' names)
    'fighters': ['scorpion', 'subzero'],      # ids from fighters/ and fighters/fan/; the first starts on the left
    'stage': 'temple',                        # temple | city | courtyard (stages/*.py)
    'start_area': 'courtyard',                # an area of the stage (python tools/kit.py stage <name> lists them)
    'length': 120,                            # seconds of video; the fight is fitted to it within a couple of seconds
    'seed': 1,                                # another seed, another fight
    'winner': 'scorpion',                     # who wins (default: the first fighter)
    'finisher': None,                         # None: the winner's own (fire_breath, ice_shatter, knockout)
    'music': 'ninja',                         # ninja (the MK-like track) | heroic (brass against a brute's riff)
    'output': 'left-vs-right',                # file names: <output>-1080p.mp4, <output>-under-30mb.mp4, ...
    'review_dir': None,                       # a folder name under the lab-review folder to copy the files into
    'auto_pins': True,                        # let the choreographer pin each kit's specials in acts you leave empty
    # pinned beats, in order, each in an act: opening, travel1, weapons, magic, travel2, escalation, turning, climax.
    # 'by': who leads it (a fighter id, 'winner' or 'loser'); other keys are the beat's parameters.
    # Special names: 'travel' with 'to': <area> (a route of the stage), 'stage_piece' with 'piece': <name> (the
    # stage's own set pieces, e.g. the temple's 'bridge_snap').
    'pins': [
        # {'act': 'opening', 'beat': 'teleport_strike', 'by': 'scorpion'},
        # {'act': 'weapons', 'beat': 'draw_weapons', 'by': 'scorpion'},
        # {'act': 'climax', 'beat': 'spear', 'by': 'scorpion', 'line': 'GET OVER HERE!'},
    ],
}
