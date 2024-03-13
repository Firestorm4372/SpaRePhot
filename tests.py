import unittest

import numpy as np

class TestData(unittest.TestCase):
    def setUp(self):
        from spare.data import Data
        self.data = Data()

    def test_size_cat_import(self):
        self.assertNotEqual(len(self.data.size_cat), 0)
    
    def test_image_segmap_same_size(self):
        segmap_size = self.data.segmap.size
        image_size =self.data.images.values[self.data.filters[0]].size
        self.assertEqual(segmap_size, image_size)


class TestGalaxy(unittest.TestCase):
    def setUp(self) -> None:
        from spare.data import Data
        from spare import galaxy
        data = Data()
        id = data.size_cat['ID'][np.random.randint(len(data.size_cat))]
        self.galaxy = galaxy.extract_galaxy(id, data)

    def test_id_in_segmap(self):
        self.assertTrue(np.isin(self.galaxy.id, self.galaxy.segmap))

if __name__ == '__main__':
    unittest.main()