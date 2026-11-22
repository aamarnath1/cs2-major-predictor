"""Singapore Major 2026 participants, seeds and Liquipedia -> VRS name mapping.

Seeds are listed in Valve Global Standings order (seed 1 first), as published on
Liquipedia (PGL/2026/Singapore). Stand-ins come from the Liquipedia roster notes.
"""

# (display name, VRS team name)
STAGE1 = [
    ("Ninjas in Pyjamas", "Ninjas in Pyjamas"),
    ("PARIVISION", "PARIVISION"),
    ("HEROIC", "HEROIC"),
    ("FaZe", "FaZe"),
    ("Liquid", "Liquid"),
    ("Eternal Fire", "Eternal Fire"),
    ("100 Thieves", "100 Thieves"),
    ("9z", "9z"),
    ("M80", "M80"),
    ("DENDELE", "DENDELE"),
    ("Imperial", "Imperial"),
    ("Wildcard", "Wildcard"),
    ("GOSH", "BC.Game"),  # s1mple / Magisk / Senzu / mzinho / asap
    ("FlyQuest", "FlyQuest"),
    ("Chinggis Warriors", "Chinggis Warriors"),
    ("Lynn Vision", "Lynn Vision"),
]

STAGE2_INVITES = [
    ("FUT", "FUT"),
    ("Nemiga", "Nemiga"),
    ("G2", "G2"),
    ("The One", "1win"),
    ("Natus Vincere", "Natus Vincere"),
    ("Astralis", "Astralis"),
    ("The MongolZ", "The MongolZ"),
    ("MIBR", "MIBR"),
]

STAGE3_INVITES = [
    ("MOUZ", "MOUZ"),
    ("Vitality", "Vitality"),
    ("FURIA", "FURIA"),
    ("Aurora", "Aurora"),
    ("Spirit", "Spirit"),
    ("BetBoom", "BETBOOM"),
    ("Falcons", "Falcons"),
    ("Legacy", "Legacy"),
]

# Players listed as not playing at the Major, replaced by a stand-in (count of swapped players).
STAND_INS = {
    "Vitality": 1,  # jL out, apEX in
    "Legacy": 1,  # n1ssim out, ksloks in
    "FUT": 1,  # cmtry out, SunPayus in
    "Astralis": 1,  # jabbi out, Anlelele in
    "9z": 1,  # meyern out, history in
}
