"""Parse and validate repeatable experiment task datasets."""

import json

from pydantic import ValidationError

from experiments.schemas import ExperimentDataset


class DatasetManager:
    def parse_json(self, raw_json: str) -> ExperimentDataset:
        try:
            data = json.loads(raw_json)
            return ExperimentDataset.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError("Dataset JSON is invalid or does not match the dataset schema") from error

    @staticmethod
    def serialize(dataset: ExperimentDataset) -> dict[str, object]:
        return dataset.model_dump(mode="json")