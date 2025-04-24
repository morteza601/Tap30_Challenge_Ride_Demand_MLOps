"""
This modules handles the transformation of raw taxi demand data into a format suitable for model training.
The processing includes time-based feature engineering to capture temporal patterns in taxi demand
and ensures consistent data organization across train, validation, and test sets.
"""

from pathlib import Path

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


class DataProcessing:

    def __init__(self, config):
        self.data_processing_config = config["data_processing"]
        artifact_dir = Path(config["data_ingestion"]["artifact_dir"])
        self.raw_dir = artifact_dir / "raw"

        self.processed_dir = artifact_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self):
        """
        Loads the split datasets.

        Returns
        -------
        tuple
            Train, validation, and test dataframes
        """

        train_data = pd.read_csv(self.raw_dir / "train.csv")
        val_data = pd.read_csv(self.raw_dir / "val.csv")
        test_data = pd.read_csv(self.raw_dir / "test.csv")

        return train_data, val_data, test_data

    def process_data(self, train_data, val_data, test_data):
        """
        Applies identical processing steps to all datasets to ensure consistent
        feature engineering across splits.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training dataset
        val_data : pd.DataFrame
            Validation dataset
        test_data : pd.DataFrame
            Test dataset

        Returns
        -------
        tuple
            Processed train, validation, and test dataframes
        """

        train_data = self._process_single_dataset(train_data)
        val_data = self._process_single_dataset(val_data)
        test_data = self._process_single_dataset(test_data)

        return train_data, val_data, test_data

    def _process_single_dataset(self, data):
        """
        Transforms raw temporal data into meaningful features that capture
        daily and hourly patterns in taxi demand, making it easier for the
        model to learn time-based patterns.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset to process

        Returns
        -------
        pd.DataFrame
            Dataset with engineered temporal features
        """

        data = data.sort_values(by=["time", "row", "col"]).reset_index(drop=True)
        data["time"] = data["time"] + self.data_processing_config["shift"]
        data = data.assign(hour_of_day=data["time"] % 24, day=data["time"] // 24)
        data = data.drop(columns=["time"])
        return data

    def save_to_csv_files(
        self, processed_train_data, processed_val_data, processed_test_data
    ):
        """
        Saves processed data.

        Parameters
        ----------
        train_data : pd.DataFrame
            Processed training dataset
        val_data : pd.DataFrame
            Processed validation dataset
        test_data : pd.DataFrame
            Processed test dataset
        """

        column_order = ["hour_of_day", "day", "row", "col", "demand"]

        train_data = processed_train_data[column_order]
        val_data = processed_val_data[column_order]
        test_data = processed_test_data[column_order]

        train_data.to_csv(self.processed_dir / "train.csv", index=False)
        val_data.to_csv(self.processed_dir / "val.csv", index=False)
        test_data.to_csv(self.processed_dir / "test.csv", index=False)

        logger.info(f"Saved processed data to {self.processed_dir}.")

    def run(self):
        """Execute the complete data ingestion pipeline.

        This method orchestrates the entire data processing process:
        1. Loads the raw data
        2. Processes the data
        3. Saves the processed data

        Examples
        --------
        >>> data_processing = DataProcessing(read_config("config/config.yaml"))
        >>> data_processing.run()
        """

        logger.info("Starting data processing...")
        train_data, val_data, test_data = self.load_raw_data()
        processed_train_data, processed_val_data, processed_test_data = (
            self.process_data(train_data, val_data, test_data)
        )
        self.save_to_csv_files(
            processed_train_data, processed_val_data, processed_test_data
        )
        logger.info("Data processing completed.")
