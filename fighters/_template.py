"""TEMPLATE for a new fighter. Copy this file to fighters/<id>.py (an original character) or fighters/fan/<id>.py (a
franchise character: private clips only), change the values, and name the id in a recipe. The file name is the id.

Check it without rendering anything:
    python tools/kit.py fighter <id>              loads it, lists problems, draws him in three views into out/kit/
    python tools/kit.py                           everything you can name here: stances, strikes, weapons, powers
                                                  (beats), finishers, head kinds, emblems, builds

Everything below is data. The engine code changes only when a fighter needs a truly new kind of move: then write a
beat in engine/beats.py (see the end of this file) and every fighter can use it.
"""

FIGHTER = {
    # ---------------------------------------------------------------- who he is
    'id': 'template',                 # = the file name; used in recipes ('fighters': ['template', 'subzero'])
    'name': 'TEMPLATE',               # on the title card and the win card ('TEMPLATE WINS')
    'short': 'TEMPLATE',              # optional shorter name
    'tier': 3,                        # power tier, 1..5: 1 a trained human (Captain America), 3 a magic warrior (the
                                      # MK ninjas), 5 a titan (the Hulk). It scales his leaps, knockback, magic and
                                      # destruction, and how far he is thrown by a stronger foe (engine/acts.power)
    'size': 1.0,                      # drawn size against a normal man (the Hulk 1.5, Captain America 1.07)
    'stance': 'ninja',                # his guard, stalk and run: 'ninja' (boxer's guard), 'shield' (shield forward,
                                      # shield block, shield flips), 'brute' (hunched, heavy steps, brute reactions)
    'title_colors': 'gold',           # the title card's word colour: fire, ice, gold, bone, blood, hero, green, red
    'trail': 'ffffff',                # afterimage tint on dashes and big moves (hex)
    'element': 'body',                # fire | ice | shield | strength | body: hands glow with it while he casts,
                                      # his entrance bursts with it

    # ---------------------------------------------------------------- how he looks
    'look': {
        'build': 'human',             # human | lean | athletic | heavy | brute (limb thickness, shoulder spread)
        'bulk': {},                   # optional per-part thickness: {'upper': 1.2, 'chest_c': 1.1, 'head': 1.0, ...}
        # materials: five-tone ramps, darkest to lightest (hex). Name them as you like and use the names below.
        'palette': {
            'suit': ('0a0a10', '181822', '262634', '3a3a4e', '58587a'),
            'trim': ('3b2204', '875608', 'c98d14', 'f0bd2c', 'ffe58a'),
            'skin': ('3a1a10', '7a3e26', 'b06a44', 'd99466', 'f2c09a'),
        },
        # each limb: bands of material along it, t from the joint nearest the body (0) to the far end (1).
        # (material, t0, t1[, pattern[, options]]); the first band is the base. patterns: 'wraps' (darker stripes
        # across), 'wraps_diag', 'rag' (a torn edge at t1); options: {'cuff': True} (a dark line where the band starts)
        'limbs': {
            'upper': [('suit', 0, 1), ('trim', 0, 0.30)],
            'fore': [('suit', 0, 1), ('trim', 0.30, 0.97, 'wraps')],
            'thigh': [('suit', 0, 1)],
            'shin': [('suit', 0, 1), ('trim', 0.40, 1.0, None, {'cuff': True})],
            'foot': [('suit', 0, 1)],
            'fist': 'suit',               # glove / bare hand material
            'deltoid': 'trim',            # the shoulder ball
            'under': 'skin',              # what a torn costume shows ('blood' for bare skin fighters)
            'marks': {},                  # lines on limbs: {'seam': ('upper',), 'veins': ('upper', 'fore'),
                                          #                  'muscle': ('upper',), 'slash': ('thigh',)}
        },
        # the torso: belly (with optional stripes), belt, chest, a V neckline, an emblem on the chest
        'torso': {
            'belly': 'trim', 'stripes': None, 'belt': 'suit', 'belt_knot': True, 'buckle': None,  # buckle: hex
            'chest': 'trim', 'vee': 'suit', 'lapel': True, 'pecs': True,   # pecs: True (light) or 'big'
            'emblem': None,               # {'kind': 'star' | 'spider' | 'ring' | 'bolt' | 'diamond', 'mat': 'suit',
                                          #  'size': 2.6}
            'neck': 'suit',
            'seam': False, 'abs': False, 'low_shade': False, 'belt_ragged': False,
        },
        # the head. kinds:
        #   'ninja'  : hood + face window + lower mask   {'hood', 'mask', 'skin', 'mask_line': hex}
        #   'helmet' : helmet with a face opening        {'helmet', 'skin', 'letter': 'A', 'letter_mat', 'wings': True}
        #   'mask'   : a full mask with big lens eyes    {'mask', 'web': hex (web lines), 'lens_rim': hex};
        #              the lens colour is 'eye' below
        #   'bare'   : a bare head with hair             {'skin', 'hair', 'angry_eye': hex}
        'head': {'kind': 'ninja', 'hood': 'trim', 'mask': 'suit', 'skin': 'skin'},
        'flaps': {'mat': 'trim', 'front': True, 'back': True},   # a tabard from the belt, or None
        'cape': None,                     # {'mat': 'trim', 'length': 1.0}: a cape from the shoulders, or None
        'gear': {},                       # {'scabbards': 2} swords on the back, {'rope': True} a coil at the hip,
                                          # {'shield': {'radius': 8.6, 'colors': (ring, ring, centre, star)}}
        'rim': 'ffffff',                  # rim light colour
        'eye': 'ffffff',                  # eyes (or a mask's lenses)
        'smear': 'ffffff',                # the streak behind fast blows
    },

    # ---------------------------------------------------------------- what he can do
    # hand to hand: move names from engine/moves.py (python tools/kit.py lists them). The choreographer strings light
    # blows, finishes with heavy ones, launches with a launcher, follows in the air.
    'strikes': {
        'light': ['jab', 'cross', 'elbow'],
        'heavy': ['hook', 'roundhouse', 'front_kick', 'knee'],
        'launcher': ['uppercut'],
        'low': ['sweep'],
        'air': ['air_kick', 'air_punch', 'axe_kick'],
        'leap': ['jump_kick'],
    },
    # weapons taken up mid-fight, in order. kind: a name in engine/body.WEAPONS (sword, ice_sword, hammer, staff,
    # axe). how: 'draw' (from his back, steel) or 'form' (made of his element). hands: 'both' (one in each hand) or
    # 'one'. heavy: True for the heavier weapon formed after the first one breaks.
    'weapons': [
        # {'kind': 'sword', 'hands': 'both', 'how': 'draw'},
    ],
    # specials: each names a beat of the shared library (engine/beats.py; tools/kit.py lists them with what they do).
    # Flags: 'defence': True (used when the other attacks: the ice wall, the shield brace), 'trap': True (the turning
    # point: the ice clone).
    'powers': [
        # {'beat': 'fire_dash', 'element': 'fire'},
        # {'beat': 'ice_ball', 'element': 'ice'},
        # {'beat': 'shield_throw', 'element': 'shield'},
    ],
    'signature': {'beat': 'big_combo'},   # the climax when he wins: spear (with 'line'), ice_spikes_launch,
                                          # shield_combo, leap_smash, big_combo
    'breakout': None,                 # how he gets out of a freeze: 'fire' (bursts out burning) or None
    'finisher': 'knockout',           # the end when he wins: fire_breath, ice_shatter (MK fatalities, with FINISH HIM
                                      # and FATALITY), knockout (a last blow, he stays down)
    'entrance': 'walk_in',            # under the title: walk_in, drop_down (flips down from up high), fire_burst
    'voice': 'announcer',             # voice style for his lines (audio/voice.STYLES: announcer, scorpion, hulk)
}

# ---------------------------------------------------------------- a new kind of move (engine code)
# When no beat does what a new fighter needs (a web line that pulls, a lightning bolt), write one in engine/beats.py:
#
#   @beat('magic', 'escalation', weight=1.4, needs=lambda c, A, D, t: has(A, 'web_pull'))
#   def web_pull(c, A, D, t):
#       """What it does, in one plain sentence."""
#       ... schedule A's and D's moves with acts.strike / A.add(...), effects from fx / magic, sound cues ...
#       return t_end
#
# Then name it in the fighter's 'powers'. Build it from what is there: acts.strike lands a blow where it should,
# acts.tumble_to and fall_lie_up throw and drop a body, magic.py has fire, ice and their bigger forms, fx.Spear is a
# rope line with a blade (spear(), spear_batted()), fx.Shield a thrown disc on waypoints. New poses go in
# engine/moves.py (a function of u returning a pose), new sound kinds in audio/sfx.py with @register.
