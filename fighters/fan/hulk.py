"""The Hulk (fan fighter, private clips only). A green titan: swats, haymakers, double-fist smashes, a thunderclap that
blasts a room, leaps that crater the road. Power tier 5 (a titan): his blows throw a man across the area and through
walls, he barely moves when hit."""

FIGHTER = {
    'id': 'hulk',
    'name': 'HULK',
    'tier': 5,
    'size': 1.5,
    'stance': 'brute',
    'title_colors': 'green',
    'trail': '72e04a',
    'element': 'strength',
    'look': {
        'build': 'brute',
        'palette': {
            'skin': ('0c2006', '1c4610', '30741e', '4ea232', '86d65a'),
            'pants': ('180626', '34104c', '541c76', '7630a0', 'a258cc'),
            'hair': ('040405', '0c0c10', '16161c', '202028', '30303a'),
        },
        'limbs': {
            'upper': [('skin', 0, 1)],
            'fore': [('skin', 0, 1)],
            'thigh': [('pants', 0, 1)],
            'shin': [('skin', 0, 1), ('pants', 0, 0.5, 'rag')],
            'foot': [('skin', 0, 1)],
            'fist': 'skin',
            'deltoid': 'skin',
            'under': 'blood',
            'marks': {'slash': ('thigh',), 'veins': ('upper', 'fore'), 'muscle': ('upper',)},
        },
        'torso': {'belly': 'skin', 'belt': 'pants', 'belt_ragged': True, 'abs': True, 'chest': 'skin', 'pecs': 'big',
                  'neck': 'skin'},
        'head': {'kind': 'bare', 'skin': 'skin', 'hair': 'hair', 'angry_eye': 'd8ff70'},
        'gear': {},
        'rim': 'c4ff96',
        'eye': 'f4fff0',
        'smear': '8ad860',
    },
    'strikes': {
        'light': ['hulk_jab', 'swat'],
        'heavy': ['haymaker', 'double_fist_high', 'swat'],
        'launcher': ['swat'],
        'low': ['stomp'],
        'air': [],
        'leap': [],
    },
    'weapons': [],
    # a titan's kit (shared beats): grabs and throws, cars and road slabs thrown, the ground smash, the thunderclap,
    # leaps across the block, the rage; plus the routes only a titan opens (the road caving in, the leap back up)
    'powers': [
        {'beat': 'ankle_slam', 'element': 'strength'},
        {'beat': 'throat_slam', 'element': 'strength'},
        {'beat': 'car_throw', 'element': 'strength'},
        {'beat': 'chunk_throw', 'element': 'strength'},
        {'beat': 'ground_smash', 'element': 'strength'},
        {'beat': 'charge_trip', 'element': 'strength'},
        {'beat': 'charge', 'element': 'strength'},
        {'beat': 'shockwave', 'element': 'strength'},
        {'beat': 'leap_smash', 'element': 'strength'},
        {'beat': 'rage', 'element': 'strength'},
    ],
    'signature': {'beat': 'leap_smash', 'line': 'HULK SMASH!'},
    'breakout': 'strength',
    'finisher': 'titan_smash',       # the Marvel ending: HULK SMASH!, the street blown apart, HULK WINS, the end card
    'entrance': 'meteor',           # drops out of the sky onto the road
    'voice': 'hulk',
}
