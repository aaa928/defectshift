"""The 20 models of the main roster (NODE and TabNet make 22): name -> family, training route, implementation name and early stopping.

route: ml = models/ml.py (+ protocol.fit_ml), transformer = models/transformer.py, tabular_dl = models/tabular_dl.py
       (pytabkit), foundation = models/foundation.py
uses_es: True for early-stopping models, which split the n* records 80/20 in ID_match and OOD_match and stop on the
         20%. False: the model trains on all n* records and receives no validation set.
NODE and TabNet (the other two models of the 22) are run through models/extra_models.py.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    name: str       # name used in the manuscript and the results
    family: str     # classical / tree / deep / foundation
    route: str
    impl: str       # model name inside the model code
    uses_es: bool


_ROWS = [
    ("LinearRegression", "classical", "ml", "LinearRegression", False),
    ("Ridge", "classical", "ml", "LinearModel", False),
    ("kNN", "classical", "ml", "kNN", False),
    ("SVR", "classical", "ml", "SVM_rbf", False),
    ("DecisionTree", "tree", "ml", "DecisionTree", False),
    ("RandomForest", "tree", "ml", "RandomForest", False),
    ("GBR", "tree", "ml", "GBR", False),
    ("XGBoost", "tree", "ml", "XGBoost", True),
    ("LightGBM", "tree", "ml", "LightGBM", True),
    ("CatBoost", "tree", "ml", "CatBoost", True),
    ("MLP", "deep", "transformer", "MLP", True),
    ("FT-Transformer", "deep", "transformer", "FT_Transformer", True),
    ("RealMLP", "deep", "tabular_dl", "RealMLP", True),
    ("TabM", "deep", "tabular_dl", "TabM", True),
    ("TabPFN", "foundation", "foundation", "TabPFN", False),
    # the five below use the wrappers of models/foundation.py; Kumo, TabDPT and EXAONE are judged by the qualification
    # check (foundation.QUALIFY_JUDGED)
    ("TabICL", "foundation", "foundation", "TabICL", False),
    ("TabFM", "foundation", "foundation", "TabFM", False),
    ("Kumo", "foundation", "foundation", "Kumo", False),
    ("TabDPT", "foundation", "foundation", "TabDPT", False),
    ("EXAONE", "foundation", "foundation", "EXAONE", False),
]
SPECS = {r[0]: Spec(*r) for r in _ROWS}
assert len(SPECS) == 20


def get(name):
    return SPECS[name]
