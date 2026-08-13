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

    def test_sliced_boxes_export(self):
        project = MagicMock(spec=models.Project)
        project.id = 1
        project.type = "Yolo"

        labels_data = [
            ("img_0.jpg", [
                (0, [0.5, 0.5, 0.4, 0.4]), # class 0 (cat)
                (1, [0.2, 0.2, 0.3, 0.3]), # class 1 (dog)
            ])
        ]

        request_opts = DatasetRequest(
            session_id="test_session",
            export_mode="crop",
            split_enabled=False,
            augmentation=None,
        )

        dummy_img = Image.new("RGB", (200, 200), color="blue")

        with unittest.mock.patch("dataset_generator.UPLOAD_DIR") as mock_upload:
            mock_session_dir = MagicMock()
            mock_upload.__truediv__.return_value.__truediv__.return_value = mock_session_dir
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_session_dir.__truediv__.return_value = mock_path

            with unittest.mock.patch("PIL.Image.open", return_value=dummy_img):
                zip_io = generate_dataset_zip(
                    project=project,
                    session_id="test_session",
                    labels_data=labels_data,
                    class_map={0: "cat", 1: "dog"},
                    options=request_opts
                )

        with zipfile.ZipFile(zip_io, 'r') as zf:
            file_list = zf.namelist()

        self.assertIn("cat/img_0_crop_1.jpg", file_list)
        self.assertIn("dog/img_0_crop_2.jpg", file_list)
        self.assertNotIn("data.yaml", file_list)

    def test_ocr_dataset_export_grayscale_and_augmentation(self):
        project = MagicMock(spec=models.Project)
        project.id = 1
        project.type = "Ocr"

        labels_data = [
            ("img_ocr.jpg", "OCR Text Value"),
        ]

        aug_opts = AugmentationOptions(
            num_augs=2,
            ocr_distortion_intensity=5.0,
            ocr_noise_intensity=3.0,
            ocr_blur_intensity=2.0,
        )

        request_opts = DatasetRequest(
            session_id="test_session_ocr",
            split_enabled=False,
            grayscale=True,
            augmentation=aug_opts,
        )

        dummy_img = Image.new("RGB", (150, 40), color="white")

        with unittest.mock.patch("dataset_generator.UPLOAD_DIR") as mock_upload:
            mock_session_dir = MagicMock()
            mock_upload.__truediv__.return_value.__truediv__.return_value = mock_session_dir
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_session_dir.__truediv__.return_value = mock_path

            with unittest.mock.patch("PIL.Image.open", return_value=dummy_img):
                zip_io = generate_dataset_zip(
                    project=project,
                    session_id="test_session_ocr",
                    labels_data=labels_data,
                    class_map={},
                    options=request_opts
                )

        with zipfile.ZipFile(zip_io, 'r') as zf:
            file_list = zf.namelist()
            self.assertIn("images/img_ocr.jpg", file_list)
            self.assertIn("labels/img_ocr.txt", file_list)
            # Should have augmented files
            self.assertIn("images/img_ocr_aug_1.jpg", file_list)
            self.assertIn("images/img_ocr_aug_2.jpg", file_list)

    def test_deskewer_dataset_export(self):
        project = MagicMock(spec=models.Project)
        project.id = 1
        project.type = "Deskewer"

        labels_data = [
            ("img_deskew.jpg", (-45, "10,20,100,200")),
        ]

        aug_opts = AugmentationOptions(
            num_augs=2,
            deskew_angles=[15, 30]
        )

        request_opts = DatasetRequest(
            session_id="test_session_deskew",
            split_enabled=False,
            augmentation=aug_opts,
        )

        dummy_img = Image.new("RGB", (200, 200), color="blue")

        with unittest.mock.patch("dataset_generator.UPLOAD_DIR") as mock_upload:
            mock_session_dir = MagicMock()
            mock_upload.__truediv__.return_value.__truediv__.return_value = mock_session_dir
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_session_dir.__truediv__.return_value = mock_path

            with unittest.mock.patch("PIL.Image.open", return_value=dummy_img):
                zip_io = generate_dataset_zip(
                    project=project,
                    session_id="test_session_deskew",
                    labels_data=labels_data,
                    class_map={},
                    options=request_opts
                )

        with zipfile.ZipFile(zip_io, 'r') as zf:
            file_list = zf.namelist()
            self.assertIn("images/img_deskew.jpg", file_list)
            self.assertNotIn("labels/img_deskew.txt", file_list)
            
            # Should have augmented files
            self.assertIn("images/img_deskew_aug_1.jpg", file_list)
            self.assertNotIn("labels/img_deskew_aug_1.txt", file_list)
            self.assertIn("images/img_deskew_aug_2.jpg", file_list)
            self.assertNotIn("labels/img_deskew_aug_2.txt", file_list)

if __name__ == '__main__':
    unittest.main()

