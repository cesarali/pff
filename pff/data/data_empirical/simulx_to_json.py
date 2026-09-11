"""
Tools for converting simulx output .csv files (simulation from an NLME model) to study JSON format
"""

import csv
from collections import defaultdict
from typing import Sequence

from pff.data.data_empirical.json_schema import StudyJSON


def simulx_to_json(
    csv_path,
    study_name="simulated_study",
    substance_name="Drug_A",
    dosing_type="oral"
) -> Sequence[StudyJSON]:
    # rep -> ID -> data
    reps = defaultdict(lambda: defaultdict(lambda: {
        "observations": [],
        "observation_times": [],
        "dosing": [],
        "dosing_type": [],
        "dosing_times": [],
        "dosing_name": []
    }))

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rep = int(row["rep"])
            id_ = row["ID"]
            time = float(row["TIME"])

            # Observations
            if row["value"] != ".":
                reps[rep][id_]["observations"].append(float(row["value"]))
                reps[rep][id_]["observation_times"].append(time)

            # Dosing (assumed at TIME == 0)
            if time == 0 and row["AMOUNT"] != ".":
                reps[rep][id_]["dosing"].append(float(row["AMOUNT"]))
                reps[rep][id_]["dosing_times"].append(0.0)
                reps[rep][id_]["dosing_type"].append(dosing_type)
                reps[rep][id_]["dosing_name"].append(dosing_type)

    # Build final output: one JSON object per rep
    output = []

    for rep, ids in sorted(reps.items()):
        contexts = []
        for i, (id_, data) in enumerate(ids.items()):
            contexts.append({
                "name_id": f"context_{id_}",
                **data
            })

        study = StudyJSON({
            "context": contexts,
            "meta_data": {
                "study_name": f"{study_name}_rep{rep}",
                "substance_name": substance_name
            }
        })
        output.append(study)

    return output

if __name__ == "__main__":
    output = simulx_to_json(csv_path="data/raw_nlme_simulx/indometacin-test-data.csv")

