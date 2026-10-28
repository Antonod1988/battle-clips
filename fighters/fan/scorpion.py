"""Scorpion (fan fighter, private clips only). A ninja of hellfire: twin swords on his back, a kunai on a rope, teleports
through his own fire. Power tier 3 (a magic warrior)."""

FIGHTER = {
    'id': 'scorpion',
    'name': 'SCORPION',
    'tier': 3,
    'size': 1.0,
    'stance': 'ninja',
    'title_colors': 'fire',
    'trail': 'ff8c26',
    'element': 'fire',
    'look': {
        'build': 'human',
        'palette': {
            'tunic': ('3b2204', '875608', 'c98d14', 'f0bd2c', 'ffe58a'),
            'suit': ('0a0a10', '181822', '262634', '3a3a4e', '58587a'),
            'skin': ('3a1a10', '7a3e26', 'b06a44', 'd99466', 'f2c09a'),
        },
        'limbs': {
            'upper': [('suit', 0, 1), ('tunic', 0, 0.30)],
            'fore': [('suit', 0, 1), ('tunic', 0.30, 0.97, 'wraps')],
            'thigh': [('suit', 0, 1)],
            'shin': [('suit', 0, 1), ('tunic', 0.10, 0.86, 'wraps_diag')],
            'foot': [('suit', 0, 1)],
            'fist': 'suit',
            'deltoid': 'tunic',
        },
        'torso': {'belly': 'tunic', 'belt': 'suit', 'belt_knot': True, 'seam': True, 'chest': 'tunic', 'vee': 'suit',
                  'lapel': True, 'pecs': True, 'neck': 'suit'},
        'head': {'kind': 'ninja', 'hood': 'tunic', 'mask': 'suit', 'skin': 'skin'},
        'flaps': {'mat': 'tunic', 'front': True, 'back': True},
        'gear': {'scabbards': 2, 'rope': True},
        'rim': 'ffd28a',
        'eye': 'ffffff',
        'smear': 'ffe58a',
    },
    # hand to hand: the choreographer strings these (light blows open, heavy ones finish a string)
    'strikes': {
        'light': ['jab', 'cross', 'elbow'],
        'heavy': ['hook', 'roundhouse', 'front_kick', 'knee'],
        'launcher': ['uppercut'],
        'low': ['sweep'],
        'air': ['air_kick', 'air_punch', 'axe_kick'],
        'leap': ['jump_kick'],
    },
    # weapons drawn mid-fight (in this order); 'how': draw (from the back) | form (out of his element)
    'weapons': [
        {'kind': 'sword', 'hands': 'both', 'how': 'draw'},
    ],
    # specials: each names a beat of the shared library (engine/beats.py) and its element
    'powers': [
        {'beat': 'teleport_strike', 'element': 'fire'},
        {'beat': 'hellfire', 'element': 'fire'},
        {'beat': 'fire_dash', 'element': 'fire'},
        {'beat': 'fire_tornado', 'element': 'fire'},
        {'beat': 'spear_batted', 'element': 'rope'},
    ],
    'signature': {'beat': 'spear', 'line': 'GET OVER HERE!'},   # the climax when he is winning
    'breakout': 'fire',                                          # how he gets out of ice
    'finisher': 'fire_breath',                                   # his fatality
    'entrance': 'fire_burst',
    'voice': 'scorpion',
}
