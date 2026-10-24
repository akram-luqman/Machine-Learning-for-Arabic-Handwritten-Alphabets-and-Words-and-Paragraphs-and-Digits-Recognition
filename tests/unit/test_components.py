# -*- coding: utf-8 -*-
import unittest
import numpy as np
from src.preprocessing.image_processor import ImagePreprocessor
from src.features.feature_extractor import ArabicFeatureExtractor


class TestArabicOCRComponents(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor(target_size=(64, 64))
        self.extractor = ArabicFeatureExtractor(target_size=(64, 64))

    def test_preprocessor_output_shape(self):
        # صورة تجريبية (أبيض وأسود)
        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        dummy_img[30:70, 40:60] = 255  # شكل حرف مستطيل أبيض
        
        processed = self.preprocessor.process(dummy_img)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (64, 64))

    def test_feature_extractor_dimension(self):
        dummy_img = np.zeros((64, 64), dtype=np.uint8)
        dummy_img[20:50, 20:50] = 255
        
        features = self.extractor.extract_features(dummy_img)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 50)
        self.assertEqual(len(features), self.extractor.feature_dimension)


if __name__ == '__main__':
    unittest.main()
