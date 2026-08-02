import unittest
import zipfile
import io
from PIL import Image
from unittest.mock import MagicMock
from dataset_generator import generate_dataset_zip
from schemas import DatasetRequest, AugmentationOptions
import models

class TestDatasetGeneratorAugmentation(unittest.TestCase):
    def test_augmentation_excluded_from_test_and_valid_sets(self):
        # Create mock project
        project = MagicMock(spec=models.Project)
        project.id = 1
        project.type = "Yolo"

        # Create dummy labels data for 10 images
        labels_data = []
        for i in range(10):
            img_name = f"img_{i}.jpg"
            labels_data.append((img_name, [(0, [0.5, 0.5, 0.2, 0.2])]))

        # Options with split enabled (70% train, 20% val, 10% test) and augmentation enabled
        aug_opts = AugmentationOptions(
            flip_h=True,
            flip_v=False,
            flip_hv=False,
            grain=False,
            noise=False,
            blur=False,
            num_augs=2
        )
        request_opts = DatasetRequest(
            session_id="test_session",
            split_enabled=True,
            train_pct=70.0,
            val_pct=20.0,
            test_pct=10.0,
            augmentation=aug_opts
        )

        # Mock Image.open so we don't need real files on disk
        dummy_img = Image.new("RGB", (100, 100), color="red")
        
        with unittest.mock.patch("dataset_generator.UPLOAD_DIR") as mock_upload:
            mock_session_dir = MagicMock()
            mock_upload.__truediv__.return_value.__truediv__.return_value = mock_session_dir
            
            # mock_path.exists() -> True
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_session_dir.__truediv__.return_value = mock_path

            with unittest.mock.patch("PIL.Image.open", return_value=dummy_img):
                zip_io = generate_dataset_zip(
                    project=project,
                    session_id="test_session",
                    labels_data=labels_data,
                    class_map={0: "cat"},
                    options=request_opts
                )

        # Inspect ZIP contents
        with zipfile.ZipFile(zip_io, 'r') as zf:
            file_list = zf.namelist()

        train_files = [f for f in file_list if f.startswith("train/")]
        valid_files = [f for f in file_list if f.startswith("valid/")]
        test_files = [f for f in file_list if f.startswith("test/")]

        # Check for any augmented file names (containing _aug_)
        aug_in_train = [f for f in train_files if "_aug_" in f]
        aug_in_valid = [f for f in valid_files if "_aug_" in f]
        aug_in_test = [f for f in test_files if "_aug_" in f]

        self.assertGreater(len(aug_in_train), 0, "Train set should contain augmented files")
        self.assertEqual(len(aug_in_valid), 0, "Valid set MUST NOT contain any augmented files")
        self.assertEqual(len(aug_in_test), 0, "Test set MUST NOT contain any augmented files")

if __name__ == '__main__':
    unittest.main()
