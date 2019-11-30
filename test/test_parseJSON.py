import unittest
import parseJSON
import numpy as np
import pandas as pd
import os
import sys

class test_functions(unittest.TestCase):
    """
    Test auxiliary functions used by jsonParser
    """
    def test_splitStates1(self):
        "Split states with start and endpoints"
        a = pd.DataFrame({'fire': [0, 1, 1, 1],
                          'clf_confidence': [1, 0, 1, 1],
                          'loc_confidence': [0, 0, 0, 0],
                          'splitEnd': None,
                          'frame': [0, 100, 200, 500]} )

        states_a = pd.DataFrame({'fire': [0, 1, 1],
                                 'clf_confidence': [1, 0, 1],
                                 'loc_confidence': [0, 0, 0],
                                 'stateStart': [0, 100, 200],
                                 'stateEnd': [99.0, 199.0, 500.0]})

        pd.testing.assert_frame_equal(parseJSON.splitStates(a), states_a)

    def test_splitStates2(self):
        "Split states without endpoint"
        b = pd.DataFrame({'fire': [0, 1, 1, 1],
                          'clf_confidence': [1, 0, 1, 0],
                          'loc_confidence': [0, 0, 0, 0],
                          'splitEnd': 700,
                          'frame': [0, 100, 200, 500]} )

        states_b = pd.DataFrame({'fire': [0, 1, 1, 1],
                                 'clf_confidence': [1, 0, 1, 0],
                                 'loc_confidence': [0, 0, 0, 0],
                                 'stateStart': [0, 100, 200, 500],
                                 'stateEnd': [99.0, 199.0, 499.0, 700.0]})

        pd.testing.assert_frame_equal(parseJSON.splitStates(b), states_b)




class test_parseJSON(unittest.TestCase):
    """
    Test parseJSON
    """
    def setUp(self):
        inputJson = os.path.join(os.path.dirname(sys.argv[0]), 'test_3_videos.json')
        inputpath = os.path.expanduser('~/Workarea/Pyronear/Wildfire')
        self.parser = parseJSON.jsonParser(inputJson, inputpath=inputpath)

    def test_files(self):
        files = '10.mp4', '19_seq0_591.mp4', '19_seq598_608.mp4'
        np.testing.assert_array_equal(self.parser.files.fname, files)

    def test_keypoints(self):
        ref_keypoints = pd.DataFrame({
         'fname': ['10.mp4']*8 + ['19_seq0_591.mp4'],
         'fire': ['1', '1', '1', '1', '0', '0', '1', '1', '1'],
         'sequence': ['0', '0', '1', '1', '2', '2', '3', '3', '0'],
         'clf_confidence': ['1', '1', '1', '1', '1', '1', '1', '0', '0'],
         'loc_confidence': ['2', '2', '0', '0', '0', '0', '2', '0', '2'],
         'x': [598.974, 609.231, 873.846, 869.744, 724.103, 543.59, 939.487, 957.949, 568.205],
         'y': [467.692, 463.59, 500.513, 506.667, 449.231, 418.462, 244.103, 237.949, 358.974],
         't': [1.32, 2.826, 15.564, 18.637, 19.779, 20.057, 28.191, 38.907, 2.261],
         'frame': [33.0, 71.0, 389.0, 466.0, 494.0, 501.0, 705.0, 973.0, 57.0]})
        keypoints = self.parser.keypoints[ref_keypoints.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(ref_keypoints, keypoints)

    def test_states(self):
        pass


if __name__ == '__main__':
    unittest.main()
