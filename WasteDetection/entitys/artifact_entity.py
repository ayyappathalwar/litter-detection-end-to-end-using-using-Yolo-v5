from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_zip_file_path: str
    feature_store_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool

@dataclass
class Model_Trainer_Artifact:
    trained_model_file_path: str
