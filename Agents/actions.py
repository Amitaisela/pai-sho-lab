"""Action-tuple helpers, so agents never unpack actions by fixed arity.

Official-rules action shapes (spec §3):
    ('plant', tile, r, c)              plant a flower / place an accent / Boat on an accent
    ('plant', 'Boat', r, c, dr, dc)    Boat on a flower; (dr, dc) is where that flower moves
    ('arrange', fr, fc, tr, tc)
    ('skip_bonus',)                    decline a Harmony Bonus
"""


def action_kind(action):
    """'plant', 'arrange' or 'skip_bonus'."""
    return action[0]


def plant_parts(action):
    """(tile, r, c, displace) for a plant action; displace is (dr, dc) or None."""
    displace = (action[4], action[5]) if len(action) == 6 else None
    return action[1], action[2], action[3], displace


def arrange_parts(action):
    """(fr, fc, tr, tc) for an arrange action."""
    return action[1], action[2], action[3], action[4]
