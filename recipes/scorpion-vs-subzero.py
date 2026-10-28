"""Version 5 of the owner's private fan clip: Scorpion against Sub-Zero in the temple (made only with make.py).

Kept from version 4 (pinned below): the teleport, the swords and the ice sword, the disarm and the hammer, the felled
pillar, the kunai batted away, hellfire against the ice wall, the roof, the ice clone and the breakout, the bridge and
its snap, the kunai through the shoulder with «GET OVER HERE!», FINISH HIM, the fire fatality, SCORPION WINS.
Everything else the choreographer builds from the two fighters' kits and the stage.
"""

RECIPE = {
    'title': 'SCORPION vs SUB-ZERO',
    'fighters': ['scorpion', 'subzero'],          # left, right
    'stage': 'temple',
    'start_area': 'courtyard',
    'length': 123,                                # seconds of video (the choreographer aims at this)
    'seed': 7,
    'winner': 'scorpion',
    'finisher': 'fire_breath',
    'music': 'ninja',
    'output': 'versus-v5',
    'review_dir': '2026-10-28 versus-v5',
    'pins': [
        {'act': 'opening', 'beat': 'teleport_strike', 'by': 'scorpion'},
        {'act': 'travel1', 'beat': 'travel', 'to': 'ledge'},
        {'act': 'travel1', 'beat': 'ground_wave', 'by': 'subzero'},
        {'act': 'travel1', 'beat': 'exchange', 'by': 'scorpion'},
        {'act': 'travel1', 'beat': 'travel', 'to': 'courtyard'},
        {'act': 'weapons', 'beat': 'draw_weapons', 'by': 'scorpion'},
        {'act': 'weapons', 'beat': 'blade_lock', 'by': 'scorpion'},
        {'act': 'weapons', 'beat': 'disarm', 'by': 'scorpion'},
        {'act': 'weapons', 'beat': 'heavy_smash', 'by': 'subzero'},
        {'act': 'weapons', 'beat': 'recover_weapon', 'by': 'scorpion'},
        {'act': 'magic', 'beat': 'spear_batted', 'by': 'scorpion'},
        {'act': 'magic', 'beat': 'hellfire', 'by': 'scorpion'},
        {'act': 'magic', 'beat': 'ice_spikes', 'by': 'subzero'},
        {'act': 'magic', 'beat': 'fire_tornado', 'by': 'scorpion'},
        {'act': 'travel2', 'beat': 'travel', 'to': 'roof'},
        {'act': 'escalation', 'beat': 'heavy_smash', 'by': 'subzero'},
        {'act': 'escalation', 'beat': 'travel', 'to': 'courtyard'},
        {'act': 'turning', 'beat': 'ice_clone', 'by': 'subzero'},
        {'act': 'climax', 'beat': 'travel', 'to': 'peak'},
        {'act': 'climax', 'beat': 'blade_exchange', 'by': 'scorpion'},
        {'act': 'climax', 'beat': 'knock_into', 'by': 'scorpion', 'kind': 'gong'},
        {'act': 'climax', 'beat': 'stage_piece', 'piece': 'bridge_snap', 'by': 'subzero'},
        {'act': 'climax', 'beat': 'spear', 'by': 'scorpion', 'line': 'GET OVER HERE!'},
    ],
}
