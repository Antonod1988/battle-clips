"""HULK-2: the Hulk against Captain America, version 2 of the owner's private fan clip, made through battle-clips.

Version 1's winner (the Hulk) and its best moments are pinned, in its order and through all four areas: MAIN ST (the
drop from the sky, Cap's opening combo, the backhand onto a car, the charge and the lamp post, the shield's first
flight, the car thrown into the pharmacy), the ankle swing through the diner's wall, the diner (the table, the
pendant cut down, the blows on the shield, the counter broken, the column, the throat and the headbutts), the
thunderclap through the other wall, 5TH AVE (the leap, Cap off a car's roof, the shield and the hydrant, the slab
on the shield, a facade), the road caving in, the subway (Cap's long string, the rage, the third rail, the steel
column), the leap back up through the hole, Cap's last stand and HULK SMASH. Bigger than version 1 where the Hulk's
power allows: facades collapse, cars are crushed, signs and lamps come down. A Marvel ending: no FINISH HIM, no
FATALITY; HULK WINS and the end card.
"""

RECIPE = {
    'title': 'HULK vs CAPTAIN AMERICA',
    'fighters': ['captain_america', 'hulk'],
    'stage': 'city',
    'start_area': 'main',
    'length': 116,
    'seed': 2,
    'winner': 'hulk',
    'universe': 'marvel',
    'finisher': 'titan_smash',
    'music': 'heroic',
    'output': 'hulk-vs-cap-v2',
    'review_dir': '2026-10-28 hulk-vs-cap-v2',
    'pins': [
        # MAIN ST
        {'act': 'opening', 'beat': 'long_combo', 'by': 'captain_america'},
        {'act': 'opening', 'beat': 'knock_into', 'by': 'hulk', 'kind': 'car'},
        {'act': 'opening', 'beat': 'charge_trip', 'by': 'hulk'},
        {'act': 'opening', 'beat': 'shield_throw', 'by': 'captain_america', 'bounces': 2},
        {'act': 'opening', 'beat': 'car_throw', 'by': 'hulk'},
        # the ankle swing, through the diner's west wall; the Hulk charges through after him
        {'act': 'travel1', 'beat': 'travel', 'to': 'diner', 'grab': True},
        # THE DINER
        {'act': 'magic', 'beat': 'table_kick', 'by': 'captain_america'},
        {'act': 'magic', 'beat': 'shield_throw', 'by': 'captain_america', 'bounces': 2},
        {'act': 'magic', 'beat': 'shield_brace', 'by': 'captain_america'},
        {'act': 'magic', 'beat': 'ground_smash', 'by': 'hulk'},
        {'act': 'magic', 'beat': 'knock_into', 'by': 'hulk', 'kind': 'column'},
        {'act': 'magic', 'beat': 'throat_slam', 'by': 'hulk'},
        # the thunderclap blows Cap through the east wall; the Hulk walks through it
        {'act': 'travel2', 'beat': 'travel', 'to': 'fifth', 'blast': True},
        # 5TH AVE
        {'act': 'escalation', 'beat': 'leap_smash', 'by': 'hulk'},
        {'act': 'escalation', 'beat': 'car_spring', 'by': 'captain_america'},
        {'act': 'escalation', 'beat': 'shield_throw', 'by': 'captain_america', 'bounces': 2},
        {'act': 'escalation', 'beat': 'chunk_throw', 'by': 'hulk'},
        {'act': 'escalation', 'beat': 'knock_into', 'by': 'hulk', 'kind': 'facade'},
        # the meteor: the road caves in, both fall into the subway
        {'act': 'escalation', 'beat': 'travel', 'to': 'subway'},
        # THE SUBWAY
        {'act': 'turning', 'beat': 'long_combo', 'by': 'captain_america'},
        {'act': 'turning', 'beat': 'rage', 'by': 'hulk'},
        {'act': 'turning', 'beat': 'stage_piece', 'piece': 'third_rail', 'by': 'hulk'},
        {'act': 'turning', 'beat': 'shield_throw', 'by': 'captain_america', 'bounces': 1},
        {'act': 'turning', 'beat': 'knock_into', 'by': 'captain_america', 'kind': 'beam'},
        # by the throat, straight up through the hole; Cap's last stand; then HULK SMASH (the finisher)
        {'act': 'climax', 'beat': 'travel', 'to': 'fifth'},
        {'act': 'climax', 'beat': 'last_stand', 'by': 'captain_america'},
    ],
}
