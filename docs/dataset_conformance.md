# Dataset conformance — Table 2 of Monteiro 2023 (ADR 0020 §8)

`conformant` matches Table 2 exactly. `approved_with_note` is a
reconciled divergence: usable for the published comparison, with the
reason recorded. `mismatch` is excluded unless the run passes
`--allow-unverified-data`, which sets `comparable=false`.

| # | Dataset | File | Paper (inst/feat/class) | Snapshot | Status | Note | SHA-256 |
|--:|---|---|---|---|---|---|---|
| 01 | Australian | Adult.arff | 690/14/2 | 690/14/2 | `conformant` |  | `41a691f09514` |
| 02 | Banana | Banana.arff | 2000/2/2 | 2000/2/2 | `conformant` |  | `0d15f4f3ebf8` |
| 03 | Blood | Blood.arff | 748/4/2 | 748/4/2 | `conformant` |  | `0cd82ac61afa` |
| 04 | CTG | CTG.arff | 2126/21/3 | 2126/21/3 | `conformant` |  | `b627829d7441` |
| 05 | Diabetes | Diabetes.arff | 766/8/2 | 766/8/2 | `conformant` |  | `fb547f66c2ce` |
| 06 | Faults | Faults.arff | 1941/27/7 | 1941/27/7 | `conformant` |  | `94690912293d` |
| 07 | German | German.arff | 1000/24/2 | 1000/24/2 | `conformant` |  | `ade243c2c9e0` |
| 08 | Haberman | Haberman.arff | 306/3/2 | 306/3/2 | `conformant` |  | `71ab3a850d69` |
| 09 | Heart | Heart.arff | 270/10/2 | 270/13/2 | `approved_with_note` | features: paper=10 actual=13 / Table 2 Features column shifted by one row (09-13): paper 10 is ILPD's attribute count; STATLOG Heart has 13 | `fe86048ac1e9` |
| 10 | ILPD | ILPD.arff | 583/34/2 | 583/10/2 | `approved_with_note` | features: paper=34 actual=10 / Table 2 Features column shifted by one row (09-13): paper 34 is Ionosphere's attribute count; UCI ILPD has 10 | `c6e17799036a` |
| 11 | Ionosphere | Ionosphere.arff | 351/16/2 | 351/34/2 | `approved_with_note` | features: paper=16 actual=34 / Table 2 Features column shifted by one row (09-13): paper 16 is Laryngeal1's attribute count; UCI Ionosphere has 34 | `56ccbe10eb0f` |
| 12 | Laryngeal1 | Laryngeal1.arff | 213/16/2 | 213/16/2 | `conformant` |  | `3e78fb5d0768` |
| 13 | Laryngeal3 | Laryngeal3.arff | 353/2/3 | 353/16/3 | `approved_with_note` | features: paper=2 actual=16 / Table 2 Features column shifted by one row (09-13): paper 2 is Lithuanian's attribute count; LKC Laryngeal3 has 16 | `eb28a4890bbb` |
| 14 | Lithuanian | Lithuanian.arff | 2000/2/2 | 2000/2/2 | `conformant` |  | `03caa886b98f` |
| 15 | Liver | Liver.arff | 345/6/2 | 345/6/2 | `conformant` |  | `218dc94e4305` |
| 16 | Mammo | Mammo.arff | 830/5/2 | 830/5/2 | `conformant` |  | `6b2c9ccf4353` |
| 17 | Monk | Monk.arff | 432/6/2 | 432/6/2 | `conformant` |  | `247cc54f18cb` |
| 18 | Phoneme | Phoneme.arff | 5404/5/2 | 5404/5/2 | `conformant` |  | `551fb60ac057` |
| 19 | P2 | P2.arff | 5000/2/2 | 2000/2/2 | `mismatch` | instances: paper=5000 actual=2000 | `2109e2fe995a` |
| 20 | Segmentation | Segmentation.arff | 2310/19/7 | 2310/19/7 | `conformant` |  | `b3affc1ab2c9` |
| 21 | Sonar | Sonar.arff | 208/60/2 | 208/60/2 | `conformant` |  | `10a6b9f4472c` |
| 22 | Thyroid | Thyroid.arff | 692/16/2 | 692/16/2 | `conformant` |  | `614bc2d754f7` |
| 23 | Vehicle | Vehicle.arff | 846/18/4 | 846/18/4 | `conformant` |  | `4ff49d7ed0d3` |
| 24 | Vertebral | Vertebral.arff | 300/6/2 | 300/6/2 | `conformant` |  | `9124a2c8712f` |
| 25 | WBC | WBC.arff | 569/30/2 | 569/30/2 | `conformant` |  | `a9193ee9bb2e` |
| 26 | WDVG | WDVG.arff | 5000/21/3 | 5000/21/3 | `conformant` |  | `ab17c7ef08c1` |
| 27 | Weaning | Weaning.arff | 302/17/2 | 302/17/2 | `conformant` |  | `ef94e880e240` |
| 28 | Wine | Wine.arff | 178/13/3 | 178/13/3 | `conformant` |  | `c7559add3b27` |

## Totals

- `approved_with_note`: 4
- `conformant`: 23
- `mismatch`: 1
