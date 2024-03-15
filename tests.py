import unittest

import numpy as np

import spare

class TestData(unittest.TestCase):
    def setUp(self):
        self.data = spare.data.Data()

    def test_size_cat_import(self):
        self.assertNotEqual(len(self.data.size_cat), 0)
    
    def test_image_segmap_same_shape(self):
        segmap_shape = self.data.segmap.shape
        image_shape = self.data.images.values[self.data.filters[0]].shape
        self.assertEqual(segmap_shape, image_shape)


class TestGalaxy(unittest.TestCase):
    def setUp(self) -> None:
        data = spare.data.Data()
        id = data.size_cat['ID'][np.random.randint(len(data.size_cat))]
        self.galaxy = spare.galaxy.extract_galaxy(id, data)

    def test_id_in_segmap(self):
        self.assertTrue(np.isin(self.galaxy.id, self.galaxy.segmap))
    
    def test_image_segmap_same_shape(self):
        filter = list(self.galaxy.values.keys())[0]
        segmap_shape = self.galaxy.segmap.shape
        image_shape = self.galaxy.values[filter].shape
        self.assertEqual(segmap_shape, image_shape)

if __name__ == '__main__':
    unittest.main()