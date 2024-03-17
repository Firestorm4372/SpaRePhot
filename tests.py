import unittest

import json
import numpy as np

import spare

class TestData(unittest.TestCase):
    def setUp(self):
        self.data = spare.filemanage.Data()

    def test_size_cat_import(self):
        self.assertNotEqual(len(self.data.size_cat), 0)
    
    def test_image_segmap_same_shape(self):
        segmap_shape = self.data.segmap.shape
        image_shape = self.data.images.values[self.data.filters[0]].shape
        self.assertEqual(segmap_shape, image_shape)


class TestGalaxy(unittest.TestCase):
    def setUp(self) -> None:
        data = spare.filemanage.Data()
        self.galaxy = spare.extract_galaxy(55733, data)
        # add in an 'unused' pixel to later test against
        self.filter_key = [k for k in self.galaxy.errors.keys()][0]
        self.galaxy.errors[self.filter_key][0,0] = -999

    def test_id_in_segmap(self):
        self.assertTrue(np.isin(self.galaxy.id, self.galaxy.segmap))
    
    def test_image_segmap_same_shape(self):
        filter = list(self.galaxy.values.keys())[0]
        segmap_shape = self.galaxy.segmap.shape
        image_shape = self.galaxy.values[filter].shape
        self.assertEqual(segmap_shape, image_shape)

    def test_replace_unused(self):
        self.galaxy.replace_unused_with_constant(-999, -9999)
        pixel = self.galaxy.errors[self.filter_key][0,0]
        self.assertEqual(-9999, pixel)


class TestOverManageAndSave(unittest.TestCase):
    def setUp(self) -> None:
        self.fm = spare.filemanage.RunManager()
        self.selection = spare.EAZYprep.SelectionGalaxies([spare.extract_galaxy(id, spare.filemanage.Data()) for id in [55733, 74977, 183348]])
        if self.fm.runs_df.shape[0] == 0:
            self.run_id_to_delete = 0
        else:
            self.run_id_to_delete = self.fm.runs_df.index[-1] + 1

    def testGoodSave(self) -> None:
        self.selection.save_selection('test')
        run_id = self.selection.run_id
        with open(f'{self.fm.run_folder(run_id)}/galaxies/0/info.txt') as f:
            id = json.load(f)['id']
        self.assertEqual(id, 55733)

    def tearDown(self) -> None:
        self.fm.delete_run(self.run_id_to_delete)

if __name__ == '__main__':
    unittest.main()