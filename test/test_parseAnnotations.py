import unittest
import parseAnnotations
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
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
                          'frame': [0, 100, 200, 500]})

        states_a = pd.DataFrame({'fire': [0, 1, 1],
                                 'clf_confidence': [1, 0, 1],
                                 'loc_confidence': [0, 0, 0],
                                 'stateStart': [0, 100, 200],
                                 'stateEnd': [99.0, 199.0, 500.0]})

        pd.testing.assert_frame_equal(parseAnnotations.splitStates(a), states_a)

    def test_splitStates2(self):
        "Split states without endpoint"
        b = pd.DataFrame({'fire': [0, 1, 1, 1],
                          'clf_confidence': [1, 0, 1, 0],
                          'loc_confidence': [0, 0, 0, 0],
                          'splitEnd': 700,
                          'frame': [0, 100, 200, 500]})

        states_b = pd.DataFrame({'fire': [0, 1, 1, 1],
                                 'clf_confidence': [1, 0, 1, 0],
                                 'loc_confidence': [0, 0, 0, 0],
                                 'stateStart': [0, 100, 200, 500],
                                 'stateEnd': [99.0, 199.0, 499.0, 700.0]})

        pd.testing.assert_frame_equal(parseAnnotations.splitStates(b), states_b)

    def test_pickFrames0(self):
        "Simple test of pickFrames"
        state = pd.Series({'stateStart': 0, 'stateEnd': 100})
        frames = pd.Series([0, 50, 100])
        pd.testing.assert_series_equal(frames, parseAnnotations.pickFrames(state, 3, random=False))

    def test_pickFrames(self):
        "Test of pickFrames for DataFrame"
        states_a = pd.DataFrame({'fire': [0, 1, 1],
                                 'clf_confidence': [1, 0, 1],
                                 'loc_confidence': [0, 0, 0],
                                 'stateStart': [0, 100, 200],
                                 'stateEnd': [99.0, 199.0, 500.0]})

        frames = pd.DataFrame([[0, 49, 99], [100, 149, 199], [200, 350, 500]])
        x = states_a.apply(partial(parseAnnotations.pickFrames, nFrames=3, random=False), axis=1)
        pd.testing.assert_frame_equal(x, frames)

    def test_getFrameLabels(self):
        states_a = pd.DataFrame({'fBase': '10.mp4',
                                 'stateStart': [0, 100, 200],
                                 'stateEnd': [99, 199, 500]})

        labels_a = pd.DataFrame({'fBase': '10.mp4',
                                 'stateStart': [0, 0, 100, 100, 200, 200],
                                 'stateEnd': [99, 99, 199, 199, 500, 500],
                                 'frame': [0, 99, 100, 199, 200, 500],
                                 'imgFile': ['10_frame0.png', '10_frame99.png',
                                             '10_frame100.png', '10_frame199.png',
                                             '10_frame200.png', '10_frame500.png']})

        labels = parseAnnotations.getFrameLabels(states_a, nFrames=2, random=False)
        # Indices are not important and are reshuffled when sorting by fBase and frame
        pd.testing.assert_frame_equal(labels.reset_index(drop=True), labels_a)

        # Test if can pick frames again from labels
        labels_again = parseAnnotations.getFrameLabels(labels, nFrames=2, random=False, from_labels=True)
        pd.testing.assert_frame_equal(labels_again.reset_index(drop=True), labels_a)

        # Test if can pick frames again from labels with different options
        labels_b = pd.DataFrame({'fBase': '10.mp4',
                                 'stateStart': [0, 0, 0, 100, 100, 100, 200, 200, 200],
                                 'stateEnd': [99, 99, 99, 199, 199, 199, 500, 500, 500],
                                 'frame': [0, 49, 99, 100, 149, 199, 200, 350, 500],
                                 'imgFile': ['10_frame0.png', '10_frame49.png',
                                             '10_frame99.png', '10_frame100.png',
                                             '10_frame149.png', '10_frame199.png',
                                             '10_frame200.png', '10_frame350.png',
                                             '10_frame500.png']})
        labels3 = parseAnnotations.getFrameLabels(labels, nFrames=3, random=False, from_labels=True)
        pd.testing.assert_frame_equal(labels3.reset_index(drop=True), labels_b)


def setupTester(cls):
    """
    Setup tester for AnnotationParser
    """
    inputJson = Path(sys.argv[0]).parent / 'test_3_videos.json'
    inputJson_only_exploitable = Path(sys.argv[0]).parent/'test_3_videos_only_exploitable.json'
    inputdir = Path('~/Workarea/Pyronear/Wildfire').expanduser()

    cls.parser = parseAnnotations.AnnotationParser(inputJson, inputdir=inputdir)
    cls.parser_only_exploitable = parseAnnotations.AnnotationParser(inputJson_only_exploitable, inputdir=inputdir)


class test_parseAnnotations(unittest.TestCase):
    """
    Test parseAnnotations
    """
    @classmethod
    def setUpClass(cls):
        "Setup only once for all tests"
        setupTester(cls)

    def test_columns_are_right(self):
        for col_name in self.parser.labels['aname']:
            if col_name != 'spatial':  # spatial is expected to be dropped during the parsing
                self.assertIn(col_name, self.parser.keypoints.columns)

        # When all video are exploitable, is the column correctly created ?
        for col_name in self.parser_only_exploitable.labels['aname']:
            if col_name != 'spatial':  # spatial is expected to be dropped during the parsing
                self.assertIn(col_name, self.parser_only_exploitable.keypoints.columns)

    def test_files(self):
        files = '10.mp4', '19_seq0_591.mp4', '19_seq598_608.mp4'
        np.testing.assert_array_equal(self.parser.files.fname, files)

    def test_keypoints(self):
        ref_keypoints = pd.DataFrame({
            'fname': ['10.mp4'] * 8 + ['19_seq0_591.mp4'],
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

    def test_writeCsv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.parser.writeCsv(tmpdirname)
            tmpdir = Path(tmpdirname)
            basename = Path(self.parser.fname).name
            keypointFile = (tmpdir / basename).with_suffix('.keypoints.csv')
            self.assertTrue(keypointFile.exists())
            statesFile = (tmpdir / basename).with_suffix('.states.csv')
            self.assertTrue(statesFile.exists())
            # TODO: test reading csv and comparing with original


if __name__ == '__main__':
    unittest.main()
