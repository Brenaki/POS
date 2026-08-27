"""Paper reference table fixtures (ADR 2020 §9).

Versioned tabular data extracted from Monteiro 2023 (Information Fusion 89),
Decision Tree tables only — the scope of `monteiro-2023-dt-compatible-v1`.

Tables:
- Table 4: measure votes (F1..LSC per dataset)
- Table 6: average generation number selected (DT only)
- Table 7: average Oracle (mean only, no std)
- Table A.10: DT MVR accuracy (mean, std)
- Table A.12: DT DCS accuracy (LCA, OLA, Rank)
- Table A.14: DT DES accuracy (KNORA-E, KNORA-U, META-DES)

Values are percentages as published. source_table identifies the provenance.
"""

from __future__ import annotations

from typing import Any

DATASET_NAMES_PAPER = [
    "Australian", "Banana", "Blood", "CTG", "Diabetes", "Faults",
    "German", "Haberman", "Heart", "ILPD", "Ionosphere", "Laryngeal1",
    "Laryngeal3", "Lithuanian", "Liver", "Mammo", "Monk", "Phoneme",
    "P2", "Segmentation", "Sonar", "Thyroid", "Vehicle", "Vertebral",
    "WBC", "WDVG", "Weaning", "Wine",
]

TABLE_4_VOTES: dict[str, dict[str, int]] = {
    "Australian": {"F1": 4, "F1v": 3, "F2": 1, "F3": 2, "F4": 0, "N1": 0, "N2": 0, "N3": 2, "N4": 5, "T1": 1, "LSC": 2},
    "Banana": {"F1": 0, "F1v": 2, "F2": 3, "F3": 4, "F4": 1, "N1": 1, "N2": 0, "N3": 3, "N4": 0, "T1": 4, "LSC": 2},
    "Blood": {"F1": 2, "F1v": 0, "F2": 5, "F3": 3, "F4": 0, "N1": 3, "N2": 1, "N3": 1, "N4": 2, "T1": 1, "LSC": 2},
    "CTG": {"F1": 4, "F1v": 3, "F2": 2, "F3": 1, "F4": 0, "N1": 1, "N2": 3, "N3": 2, "N4": 1, "T1": 2, "LSC": 1},
    "Diabetes": {"F1": 1, "F1v": 1, "F2": 2, "F3": 5, "F4": 1, "N1": 1, "N2": 3, "N3": 1, "N4": 0, "T1": 1, "LSC": 4},
    "Faults": {"F1": 0, "F1v": 10, "F2": 0, "F3": 0, "F4": 0, "N1": 1, "N2": 1, "N3": 4, "N4": 1, "T1": 1, "LSC": 2},
    "German": {"F1": 0, "F1v": 0, "F2": 9, "F3": 1, "F4": 0, "N1": 3, "N2": 1, "N3": 0, "N4": 4, "T1": 1, "LSC": 1},
    "Haberman": {"F1": 2, "F1v": 2, "F2": 4, "F3": 0, "F4": 2, "N1": 0, "N2": 1, "N3": 3, "N4": 4, "T1": 1, "LSC": 1},
    "Heart": {"F1": 0, "F1v": 0, "F2": 0, "F3": 0, "F4": 10, "N1": 0, "N2": 0, "N3": 0, "N4": 6, "T1": 3, "LSC": 1},
    "ILPD": {"F1": 1, "F1v": 3, "F2": 0, "F3": 5, "F4": 1, "N1": 4, "N2": 2, "N3": 2, "N4": 1, "T1": 0, "LSC": 1},
    "Ionosphere": {"F1": 7, "F1v": 2, "F2": 0, "F3": 1, "F4": 0, "N1": 0, "N2": 1, "N3": 3, "N4": 5, "T1": 1, "LSC": 0},
    "Laryngeal1": {"F1": 1, "F1v": 0, "F2": 0, "F3": 1, "F4": 8, "N1": 0, "N2": 2, "N3": 2, "N4": 4, "T1": 0, "LSC": 2},
    "Laryngeal3": {"F1": 0, "F1v": 3, "F2": 0, "F3": 2, "F4": 5, "N1": 2, "N2": 4, "N3": 2, "N4": 1, "T1": 0, "LSC": 1},
    "Lithuanian": {"F1": 1, "F1v": 0, "F2": 2, "F3": 4, "F4": 3, "N1": 2, "N2": 2, "N3": 2, "N4": 0, "T1": 3, "LSC": 1},
    "Liver": {"F1": 1, "F1v": 5, "F2": 2, "F3": 0, "F4": 2, "N1": 2, "N2": 1, "N3": 3, "N4": 2, "T1": 2, "LSC": 0},
    "Mammo": {"F1": 1, "F1v": 2, "F2": 7, "F3": 0, "F4": 0, "N1": 2, "N2": 1, "N3": 1, "N4": 3, "T1": 2, "LSC": 1},
    "Monk": {"F1": 0, "F1v": 0, "F2": 4, "F3": 0, "F4": 6, "N1": 1, "N2": 4, "N3": 1, "N4": 2, "T1": 2, "LSC": 0},
    "Phoneme": {"F1": 1, "F1v": 4, "F2": 1, "F3": 2, "F4": 2, "N1": 1, "N2": 3, "N3": 4, "N4": 1, "T1": 0, "LSC": 1},
    "P2": {"F1": 2, "F1v": 5, "F2": 3, "F3": 0, "F4": 0, "N1": 1, "N2": 2, "N3": 4, "N4": 2, "T1": 0, "LSC": 1},
    "Segmentation": {"F1": 0, "F1v": 10, "F2": 0, "F3": 0, "F4": 0, "N1": 0, "N2": 2, "N3": 4, "N4": 1, "T1": 2, "LSC": 1},
    "Sonar": {"F1": 5, "F1v": 2, "F2": 0, "F3": 1, "F4": 2, "N1": 2, "N2": 0, "N3": 2, "N4": 5, "T1": 1, "LSC": 0},
    "Thyroid": {"F1": 2, "F1v": 0, "F2": 0, "F3": 5, "F4": 3, "N1": 0, "N2": 2, "N3": 2, "N4": 1, "T1": 0, "LSC": 5},
    "Vehicle": {"F1": 0, "F1v": 0, "F2": 0, "F3": 1, "F4": 9, "N1": 1, "N2": 0, "N3": 3, "N4": 0, "T1": 4, "LSC": 2},
    "Vertebral": {"F1": 1, "F1v": 2, "F2": 1, "F3": 0, "F4": 6, "N1": 2, "N2": 4, "N3": 2, "N4": 2, "T1": 0, "LSC": 0},
    "WBC": {"F1": 3, "F1v": 2, "F2": 0, "F3": 5, "F4": 0, "N1": 1, "N2": 2, "N3": 0, "N4": 3, "T1": 2, "LSC": 2},
    "WDVG": {"F1": 5, "F1v": 0, "F2": 0, "F3": 2, "F4": 3, "N1": 3, "N2": 1, "N3": 2, "N4": 0, "T1": 4, "LSC": 0},
    "Weaning": {"F1": 3, "F1v": 3, "F2": 0, "F3": 4, "F4": 0, "N1": 0, "N2": 2, "N3": 0, "N4": 6, "T1": 0, "LSC": 2},
    "Wine": {"F1": 1, "F1v": 1, "F2": 0, "F3": 8, "F4": 0, "N1": 4, "N2": 0, "N3": 5, "N4": 0, "T1": 1, "LSC": 0},
}

# Table 6: DT only (ADR 0020 scope). Format: mean (std).
TABLE_6_GENERATIONS: dict[str, dict[str, float]] = {
    "Australian": {"dt_pgdcs_a_mean": 10.5, "dt_pgdcs_a_std": 1.1, "dt_pgdcs_d_mean": 14.6, "dt_pgdcs_d_std": 3.3},
    "Banana": {"dt_pgdcs_a_mean": 8.0, "dt_pgdcs_a_std": 0.0, "dt_pgdcs_d_mean": 7.8, "dt_pgdcs_d_std": 0.7},
    "Blood": {"dt_pgdcs_a_mean": 6.5, "dt_pgdcs_a_std": 1.7, "dt_pgdcs_d_mean": 11.8, "dt_pgdcs_d_std": 4.1},
    "CTG": {"dt_pgdcs_a_mean": 9.2, "dt_pgdcs_a_std": 4.8, "dt_pgdcs_d_mean": 11.0, "dt_pgdcs_d_std": 2.0},
    "Diabetes": {"dt_pgdcs_a_mean": 12.5, "dt_pgdcs_a_std": 3.9, "dt_pgdcs_d_mean": 12.0, "dt_pgdcs_d_std": 3.7},
    "Faults": {"dt_pgdcs_a_mean": 9.5, "dt_pgdcs_a_std": 2.7, "dt_pgdcs_d_mean": 13.6, "dt_pgdcs_d_std": 0.5},
    "German": {"dt_pgdcs_a_mean": 10.6, "dt_pgdcs_a_std": 3.7, "dt_pgdcs_d_mean": 12.8, "dt_pgdcs_d_std": 2.3},
    "Haberman": {"dt_pgdcs_a_mean": 5.3, "dt_pgdcs_a_std": 0.7, "dt_pgdcs_d_mean": 14.5, "dt_pgdcs_d_std": 2.2},
    "Heart": {"dt_pgdcs_a_mean": 7.2, "dt_pgdcs_a_std": 3.2, "dt_pgdcs_d_mean": 8.8, "dt_pgdcs_d_std": 1.6},
    "ILPD": {"dt_pgdcs_a_mean": 13.4, "dt_pgdcs_a_std": 1.3, "dt_pgdcs_d_mean": 17.4, "dt_pgdcs_d_std": 2.8},
    "Ionosphere": {"dt_pgdcs_a_mean": 9.9, "dt_pgdcs_a_std": 3.3, "dt_pgdcs_d_mean": 11.0, "dt_pgdcs_d_std": 3.9},
    "Laryngeal1": {"dt_pgdcs_a_mean": 11.6, "dt_pgdcs_a_std": 4.5, "dt_pgdcs_d_mean": 12.5, "dt_pgdcs_d_std": 5.0},
    "Laryngeal3": {"dt_pgdcs_a_mean": 11.3, "dt_pgdcs_a_std": 3.5, "dt_pgdcs_d_mean": 12.5, "dt_pgdcs_d_std": 2.3},
    "Lithuanian": {"dt_pgdcs_a_mean": 7.5, "dt_pgdcs_a_std": 2.2, "dt_pgdcs_d_mean": 11.9, "dt_pgdcs_d_std": 4.3},
    "Liver": {"dt_pgdcs_a_mean": 11.4, "dt_pgdcs_a_std": 3.6, "dt_pgdcs_d_mean": 11.1, "dt_pgdcs_d_std": 5.6},
    "Mammo": {"dt_pgdcs_a_mean": 12.2, "dt_pgdcs_a_std": 1.9, "dt_pgdcs_d_mean": 10.7, "dt_pgdcs_d_std": 4.0},
    "Monk": {"dt_pgdcs_a_mean": 13.7, "dt_pgdcs_a_std": 0.7, "dt_pgdcs_d_mean": 11.6, "dt_pgdcs_d_std": 4.4},
    "Phoneme": {"dt_pgdcs_a_mean": 11.1, "dt_pgdcs_a_std": 5.2, "dt_pgdcs_d_mean": 10.9, "dt_pgdcs_d_std": 3.8},
    "P2": {"dt_pgdcs_a_mean": 9.0, "dt_pgdcs_a_std": 3.1, "dt_pgdcs_d_mean": 10.9, "dt_pgdcs_d_std": 3.8},
    "Segmentation": {"dt_pgdcs_a_mean": 7.0, "dt_pgdcs_a_std": 2.3, "dt_pgdcs_d_mean": 12.0, "dt_pgdcs_d_std": 4.4},
    "Sonar": {"dt_pgdcs_a_mean": 15.3, "dt_pgdcs_a_std": 4.9, "dt_pgdcs_d_mean": 8.7, "dt_pgdcs_d_std": 3.1},
    "Thyroid": {"dt_pgdcs_a_mean": 12.8, "dt_pgdcs_a_std": 0.9, "dt_pgdcs_d_mean": 11.6, "dt_pgdcs_d_std": 1.1},
    "Vehicle": {"dt_pgdcs_a_mean": 11.2, "dt_pgdcs_a_std": 3.2, "dt_pgdcs_d_mean": 8.4, "dt_pgdcs_d_std": 4.1},
    "Vertebral": {"dt_pgdcs_a_mean": 12.1, "dt_pgdcs_a_std": 3.1, "dt_pgdcs_d_mean": 12.5, "dt_pgdcs_d_std": 1.7},
    "WBC": {"dt_pgdcs_a_mean": 8.2, "dt_pgdcs_a_std": 2.6, "dt_pgdcs_d_mean": 12.4, "dt_pgdcs_d_std": 1.5},
    "WDVG": {"dt_pgdcs_a_mean": 11.6, "dt_pgdcs_a_std": 3.0, "dt_pgdcs_d_mean": 11.3, "dt_pgdcs_d_std": 2.6},
    "Weaning": {"dt_pgdcs_a_mean": 18.1, "dt_pgdcs_a_std": 4.8, "dt_pgdcs_d_mean": 11.9, "dt_pgdcs_d_std": 1.6},
    "Wine": {"dt_pgdcs_a_mean": 14.8, "dt_pgdcs_a_std": 5.6, "dt_pgdcs_d_mean": 12.8, "dt_pgdcs_d_std": 1.9},
}

# Table 7: DT only, mean (no std published).
TABLE_7_ORACLE: dict[str, dict[str, float]] = {
    "Australian": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Banana": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Blood": {"dt_pgdcs_a": 97, "dt_pgdcs_d": 98, "dt_bagging": 98},
    "CTG": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "Diabetes": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "Faults": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "German": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Haberman": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "Heart": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "ILPD": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "Ionosphere": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Laryngeal1": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "Laryngeal3": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Lithuanian": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Liver": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 100},
    "Mammo": {"dt_pgdcs_a": 98, "dt_pgdcs_d": 98, "dt_bagging": 97},
    "Monk": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 100},
    "Phoneme": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "P2": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 100},
    "Segmentation": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Sonar": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 100},
    "Thyroid": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "Vehicle": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 100},
    "Vertebral": {"dt_pgdcs_a": 99, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "WBC": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 99, "dt_bagging": 99},
    "WDVG": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 100},
    "Weaning": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 99},
    "Wine": {"dt_pgdcs_a": 100, "dt_pgdcs_d": 100, "dt_bagging": 100},
}

# Table A.10: DT MVR. Values are percentages.
TABLE_A10_MVR: dict[str, dict[str, float]] = {
    "Australian": {"pgdcs_a_mean": 93.0, "pgdcs_a_std": 2.3, "pgdcs_d_mean": 91.3, "pgdcs_d_std": 2.8, "bagging_mean": 87.6, "bagging_std": 1.6},
    "Banana": {"pgdcs_a_mean": 97.8, "pgdcs_a_std": 0.7, "pgdcs_d_mean": 97.6, "pgdcs_d_std": 0.7, "bagging_mean": 97.0, "bagging_std": 0.8},
    "Blood": {"pgdcs_a_mean": 83.6, "pgdcs_a_std": 2.4, "pgdcs_d_mean": 81.7, "pgdcs_d_std": 2.9, "bagging_mean": 77.4, "bagging_std": 3.2},
    "CTG": {"pgdcs_a_mean": 95.8, "pgdcs_a_std": 1.0, "pgdcs_d_mean": 95.5, "pgdcs_d_std": 1.0, "bagging_mean": 93.1, "bagging_std": 1.1},
    "Diabetes": {"pgdcs_a_mean": 85.2, "pgdcs_a_std": 2.8, "pgdcs_d_mean": 81.8, "pgdcs_d_std": 4.6, "bagging_mean": 75.8, "bagging_std": 2.8},
    "Faults": {"pgdcs_a_mean": 83.6, "pgdcs_a_std": 5.0, "pgdcs_d_mean": 84.6, "pgdcs_d_std": 2.9, "bagging_mean": 75.4, "bagging_std": 1.7},
    "German": {"pgdcs_a_mean": 85.9, "pgdcs_a_std": 5.1, "pgdcs_d_mean": 82.5, "pgdcs_d_std": 5.2, "bagging_mean": 74.9, "bagging_std": 2.7},
    "Haberman": {"pgdcs_a_mean": 80.5, "pgdcs_a_std": 6.1, "pgdcs_d_mean": 79.5, "pgdcs_d_std": 5.2, "bagging_mean": 70.4, "bagging_std": 4.1},
    "Heart": {"pgdcs_a_mean": 90.1, "pgdcs_a_std": 4.1, "pgdcs_d_mean": 88.4, "pgdcs_d_std": 5.2, "bagging_mean": 80.4, "bagging_std": 4.5},
    "ILPD": {"pgdcs_a_mean": 81.6, "pgdcs_a_std": 5.6, "pgdcs_d_mean": 81.7, "pgdcs_d_std": 4.8, "bagging_mean": 69.9, "bagging_std": 3.2},
    "Ionosphere": {"pgdcs_a_mean": 96.2, "pgdcs_a_std": 1.5, "pgdcs_d_mean": 95.9, "pgdcs_d_std": 1.9, "bagging_mean": 92.4, "bagging_std": 2.7},
    "Laryngeal1": {"pgdcs_a_mean": 91.5, "pgdcs_a_std": 4.6, "pgdcs_d_mean": 88.4, "pgdcs_d_std": 5.1, "bagging_mean": 83.1, "bagging_std": 4.7},
    "Laryngeal3": {"pgdcs_a_mean": 81.0, "pgdcs_a_std": 5.5, "pgdcs_d_mean": 81.1, "pgdcs_d_std": 4.9, "bagging_mean": 73.3, "bagging_std": 3.1},
    "Lithuanian": {"pgdcs_a_mean": 98.0, "pgdcs_a_std": 0.8, "pgdcs_d_mean": 97.8, "pgdcs_d_std": 0.5, "bagging_mean": 97.0, "bagging_std": 0.6},
    "Liver": {"pgdcs_a_mean": 85.1, "pgdcs_a_std": 3.2, "pgdcs_d_mean": 79.7, "pgdcs_d_std": 7.5, "bagging_mean": 69.4, "bagging_std": 3.7},
    "Mammo": {"pgdcs_a_mean": 85.4, "pgdcs_a_std": 4.1, "pgdcs_d_mean": 84.4, "pgdcs_d_std": 3.9, "bagging_mean": 81.5, "bagging_std": 1.8},
    "Monk": {"pgdcs_a_mean": 100.0, "pgdcs_a_std": 0.0, "pgdcs_d_mean": 99.9, "pgdcs_d_std": 0.4, "bagging_mean": 100.0, "bagging_std": 0.2},
    "Phoneme": {"pgdcs_a_mean": 96.8, "pgdcs_a_std": 1.2, "pgdcs_d_mean": 95.5, "pgdcs_d_std": 1.2, "bagging_mean": 93.1, "bagging_std": 1.0},
    "P2": {"pgdcs_a_mean": 91.9, "pgdcs_a_std": 2.6, "pgdcs_d_mean": 91.7, "pgdcs_d_std": 2.2, "bagging_mean": 88.0, "bagging_std": 0.9},
    "Segmentation": {"pgdcs_a_mean": 98.1, "pgdcs_a_std": 0.9, "pgdcs_d_mean": 97.9, "pgdcs_d_std": 0.8, "bagging_mean": 96.3, "bagging_std": 0.9},
    "Sonar": {"pgdcs_a_mean": 84.7, "pgdcs_a_std": 5.5, "pgdcs_d_mean": 85.5, "pgdcs_d_std": 6.6, "bagging_mean": 74.3, "bagging_std": 5.4},
    "Thyroid": {"pgdcs_a_mean": 97.7, "pgdcs_a_std": 1.2, "pgdcs_d_mean": 97.8, "pgdcs_d_std": 0.9, "bagging_mean": 95.9, "bagging_std": 1.0},
    "Vehicle": {"pgdcs_a_mean": 86.1, "pgdcs_a_std": 3.4, "pgdcs_d_mean": 85.2, "pgdcs_d_std": 5.1, "bagging_mean": 72.9, "bagging_std": 1.8},
    "Vertebral": {"pgdcs_a_mean": 92.1, "pgdcs_a_std": 4.1, "pgdcs_d_mean": 92.5, "pgdcs_d_std": 4.0, "bagging_mean": 85.7, "bagging_std": 3.3},
    "WBC": {"pgdcs_a_mean": 97.0, "pgdcs_a_std": 1.7, "pgdcs_d_mean": 96.6, "pgdcs_d_std": 1.9, "bagging_mean": 94.7, "bagging_std": 1.8},
    "WDVG": {"pgdcs_a_mean": 90.5, "pgdcs_a_std": 2.5, "pgdcs_d_mean": 89.5, "pgdcs_d_std": 3.0, "bagging_mean": 84.3, "bagging_std": 0.9},
    "Weaning": {"pgdcs_a_mean": 93.7, "pgdcs_a_std": 3.4, "pgdcs_d_mean": 90.1, "pgdcs_d_std": 3.0, "bagging_mean": 85.7, "bagging_std": 4.0},
    "Wine": {"pgdcs_a_mean": 99.3, "pgdcs_a_std": 1.0, "pgdcs_d_mean": 96.4, "pgdcs_d_std": 2.2, "bagging_mean": 95.2, "bagging_std": 4.4},
}

