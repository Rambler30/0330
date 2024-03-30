INV_OBJECT_LABEL ={
    0: "Stadium seats",
    1: "Healthy grass",
    2: "Stressed grass",
    3: "Artificial turf",
    4: "Evergreen trees",
    5: "Deciduous trees",
    6: "Bare earth",
    7: "Water",
    8: "Residential buildings",
    9: "Non-residential buildings",
    10: "Roads",
    11: "Sidewalks",
    12: "Crosswalks",
    13: "Major thoroughfares",
    14: "Highways",
    15: "Railways",
    16: "Paved parking lots",
    17: "Unpaved parking lots",
    18: "Cars",
    19: "Trains",
    20: "Unclassified0",
    21: "Unclassified1",
    22: "Unclassified2",
    23: "Unclassified3",
    24: "Unclassified4"
}

NORMAL_OBJECT_LABEL ={
    0: "unlabel",
    1: "Healthy grass",
    2: "Stressed grass",
    3: "Artificial turf",
    4: "Evergreen trees",
    5: "Deciduous trees",
    6: "Bare earth",
    7: "Water",
    8: "Residential buildings",
    9: "Non-residential buildings",
    10: "Roads",
    11: "Sidewalks",
    12: "Crosswalks",
    13: "Major thoroughfares",
    14: "Highways",
    15: "Railways",
    16: "Paved parking lots",
    17: "Unpaved parking lots",
    18: "Cars",
    19: "Trains",
    20: "Stadium seats"
}

CLASS_NAME = [INV_OBJECT_LABEL[i] for i in range(len(INV_OBJECT_LABEL))] + ['ignored']
NUM_BLOCKS = 64