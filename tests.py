import unittest

import numpy as np

class TestData(unittest.TestCase):
    def setUp(self):
        from spare.data import Data
        self.data = Data()

    def test_size_cat_import(self):
        self.assertNotEqual(len(self.data.size_cat), 0)
    
    def test_image_segmap_same_shape(self):
        segmap_shape = self.data.segmap.shape
        image_shape = self.data.images.values[self.data.filters[0]].shape
        self.assertEqual(segmap_shape, image_shape)


class TestGalaxy(unittest.TestCase):
    def setUp(self) -> None:
        from spare.data import Data
        from spare import galaxy
        data = Data()
        id = data.size_cat['ID'][np.random.randint(len(data.size_cat))]
        self.galaxy = galaxy.extract_galaxy(id, data)

    def test_id_in_segmap(self):
        self.assertTrue(np.isin(self.galaxy.id, self.galaxy.segmap))
    
    def test_image_segmap_same_shape(self):
        filter = list(self.galaxy.values.keys())[0]
        segmap_shape = self.galaxy.segmap.shape
        image_shape = self.galaxy.values[filter].shape
        self.assertEqual(segmap_shape, image_shape)

    def test_pixel_id_math(self):
        x = np.random.randint(self.galaxy.shape[1])
        y = np.random.randint(self.galaxy.shape[0])
        pixel_id = self.galaxy.pixel_ids[y, x]
        self.assertEqual(pixel_id, self.galaxy.pixels[pixel_id].id)

if __name__ == '__main__':
    unittest.main()