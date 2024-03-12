import unittest

class TestData(unittest.TestCase):
    def setUp(self):
        from spare.data import Data
        self.data = Data()

    def test_size_cat_import(self):
        self.assertFalse(len(self.data.size_cat), 0)
    
    def test_image_segmap_same_size(self):
        segmap_size = self.data.segmap.size
        image_size =self.data.images.values[self.data.filters[0]].size
        self.assertTrue(segmap_size, image_size)

if __name__ == '__main__':
    unittest.main()