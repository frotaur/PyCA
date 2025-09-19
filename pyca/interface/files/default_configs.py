"""
    Just a DEFAULTS dictionary with optional default parameters for each automaton.
    When an automaton is loaded, these parameters will be added in addition to the
    required (h,w) size, and device parameters.
"""


DEFAULTS = {
    "Lenia" : {
        "interest_files": "./demo_data/lenia_cool_params",
        "save_dir": "./data/lenia_saved_params"
    },
    "MaCELenia" :{
        "interest_files": "./demo_data/macelenia_cool_params",
        "save_dir": "./data/macelenia_saved_params"
    },
    "NeuralCA" : {
        "models_folder": 'saved_models/NCA'
    },
    "VonNeumann" : {
        "element_size": 9
    },
    "TotalisticCA1D" : {
        "rule": 1203,
        "k": 3,
        "r": 3
    },
}