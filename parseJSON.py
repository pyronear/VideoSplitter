import json
import numpy as np
import pandas as pd
import cv2
import os
from functools import partial

def getFps(fname, inputpath='.'):
    "Return the number of frames per second for the given movie file"
    return cv2.VideoCapture(os.path.join(inputpath, fname)).get(cv2.CAP_PROP_FPS)

def getFileInfo(fname, pattern = r'(?P<fBase>\w+)_seq(?P<splitStart>\d+)_(?P<splitEnd>\d+).(?P<ext>\w+)', inputpath='.'):
    """
    Return a DataFrame with file info from the given series containing fname

    Args:
    - fname: Series
    - pattern: fname pattern to extract fBase, splitStart, splitEnd
    - inputpath: str, where to find the movie files (default: '')

    Returns: DataFrame with columns
    - fBase (fname without _seqX_Y),
    - splitStart and splitEnd (first and last frame for split file)
    - fps (frames per second)
    """
    d = fname.str.extract(pattern).astype({'splitStart': float, 'splitEnd': float}) # to allow NaN
    d['fBase'] = (d.fBase + '.' + d.ext).fillna(fname)
    d['fps'] = d.fBase.apply(partial(getFps, inputpath=inputpath))
    return d[['fBase', 'fps', 'splitStart', 'splitEnd']]


def splitStates(df, stateKeys = ['fire', 'clf_confidence', 'loc_confidence']):
    """
    Return a DataFrame with one row per state, containing the first and last frames
    (stateStart and stateEnd) in addition to the columns in the given DataFrame
    (that must include information about the state).
    """
    def sameState(x, y):
        "Return true if rows x and y have the same state"
        return np.all(x[stateKeys] == y[stateKeys])

    # Take stateEnd as the frame of the next row - 1. For the last row it will be NaN
    # If state is the same for the last 2 rows (endpoint of sequence), add 1 to stateEnd (last frame)
    # Otherwise there was no endpoint in the sequence, set it to splitEnd
    # Finally, drop the last row if it remains at NaN (endpoint or no splitEnd defined)
    Next = df.shift(-1)
    Prev = df.shift(1) # NaN in case of 1 row in df
    states = df.rename(columns={'frame': 'stateStart'}).join(Next.frame.rename('stateEnd') - 1)
    idxLast = states.iloc[-1].name
    if not sameState(states.iloc[-1], Prev.iloc[-1]):
        states.loc[states.index[-1], 'stateEnd'] = states.splitEnd.iloc[-1]
    else:
        states.loc[states.index[-2], 'stateEnd'] += 1
    states.dropna(subset=['stateEnd'], inplace=True)
    np.testing.assert_array_less(states.stateStart, states.stateEnd)
    return states.drop(columns=['splitStart', 'splitEnd'], errors='ignore')

class jsonParser:
    """
    Parse JSON file containing annotations for movies and produce the DataFrames described
    and illustrated below.

    Args:
    - fname: str, json file
    - inputpath: str, path of movie files (default: '.')
    - defineStates: bool, define states from keypoints (default: True)

    Attributes:
    - labels: description of the information used in the annotations

    - files:
            fid 	fname 	fBase 	fps 	splitStart 	splitEnd
    0 	1 	10.mp4 	10.mp4 	25.0 	NaN 	NaN
    1 	2 	19_seq0_591.mp4 	19.mp4 	25.0 	0.0 	591.0
    2 	3 	19_seq598_608.mp4 	19.mp4 	25.0 	598.0 	608.0

    - keypoints:
        fname 	fBase 	fps 	splitStart 	splitEnd 	fire 	sequence 	clf_confidence 	loc_confidence 	exploitable 	x 	y 	t 	frame
    1 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	0 	1 	2 	True 	598.974 	467.692 	1.320 	33.0
    2 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	0 	1 	2 	True 	609.231 	463.590 	2.826 	71.0
    3 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	1 	1 	0 	True 	873.846 	500.513 	15.564 	389.0
    4 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	1 	1 	0 	True 	869.744 	506.667 	18.637 	466.0
    6 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	0 	2 	1 	0 	True 	724.103 	449.231 	19.779 	494.0
    5 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	0 	2 	1 	0 	True 	543.590 	418.462 	20.057 	501.0
    7 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	3 	1 	2 	True 	939.487 	244.103 	28.191 	705.0
    8 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	3 	0 	0 	True 	957.949 	237.949 	38.907 	973.0
    10 	19_seq0_591.mp4 	19.mp4 	25.0 	0.0 	591.0 	1 	0 	0 	2 	True 	568.205 	358.974 	2.261 	57.0

    - states:
            fname 	fBase 	fps 	fire 	sequence 	clf_confidence 	loc_confidence 	exploitable 	x 	y 	t 	stateStart 	stateEnd
    fname 	sequence
    10.mp4 	0 	1 	10.mp4 	10.mp4 	25.0 	1 	0 	1 	2 	True 	598.974 	467.692 	1.320 	33.0 	71.0
    1 	3 	10.mp4 	10.mp4 	25.0 	1 	1 	1 	0 	True 	873.846 	500.513 	15.564 	389.0 	466.0
    2 	6 	10.mp4 	10.mp4 	25.0 	0 	2 	1 	0 	True 	724.103 	449.231 	19.779 	494.0 	501.0
    3 	7 	10.mp4 	10.mp4 	25.0 	1 	3 	1 	2 	True 	939.487 	244.103 	28.191 	705.0 	972.0
    19_seq0_591.mp4 	0 	10 	19_seq0_591.mp4 	19.mp4 	25.0 	1 	0 	0 	2 	True 	568.205 	358.974 	2.261 	57.0 	591.0

    """
    def __init__(self, fname, inputpath='.', defineStates=True):
        self.inputpath = inputpath
        assert os.path.isdir(inputpath), f'Invalid path: {inputpath}'
        with open(fname) as jsonFile:
            info = json.load(jsonFile)

        # Annotation labels
        self.labels = pd.DataFrame(info['attribute'].values())
        self.labels['class'] = info['attribute'].keys()

        # Annotations
        self.annotations = pd.DataFrame(info['metadata'].values()).drop(columns='flg')

        # DataFrame with fid and fname. Add fBase, fps, splitStart, splitEnd
        # only for files with annotations
        self.files = pd.DataFrame(info['file'].values())[['fid', 'fname']]
        fnames = self.files.loc[self.files.fid.isin(self.annotations.vid), 'fname']
        self.files = self.files.join( getFileInfo(fnames, inputpath=self.inputpath) )
        for fname in self.files.fBase.dropna():
            assert os.path.isfile(os.path.join(inputpath, fname)), f'File {fname} not found in path {inputpath}'


        # Process and cleanup annotations to extract keypoints
        # - merge with 'files' to get fname, fBase, fps, splitStart, splitEnd
        # - add information from all other DataFrames (in DFS)
        # - drop columns which are not needed after processing
        # - drop lines where both fire and exploitable are NaN (no annotation)
        # - replace exploitable=NaN by True
        # - sort by fname and t
        # - drop non-exploitable
        d = pd.merge(self.annotations, self.files, left_on='vid', right_on='fid')

        def splitKeypointValues(x):
            "Convert annotation info to a Series with the keys and values"
            return pd.Series(x).rename(index=dict(zip(self.labels['class'], self.labels['aname'])))

        DFS = [
               d['av'].apply(splitKeypointValues), # annotation info
               pd.DataFrame(d.xy.tolist(), columns=['dummy', 'x', 'y']), # split xy
               pd.DataFrame(d.z.tolist(), columns=['t']), # time of keypoint
              ]
        d = d.join(DFS).drop(columns=['fid', 'xy', 'av', 'dummy', 'vid', 'z', 'spatial'])\
             .dropna(how='all', subset=['fire', 'exploitable'])\
             .fillna({'exploitable': True}).sort_values(['fname', 't'])

        # Convert time to frame
        d['frame'] = np.round(d.fps * d.t) + d.splitStart.fillna(0)
        self.keypoints = d.loc[d.exploitable != '0']

        # Define states from keypoints
        if defineStates:
            self.states = self.keypoints.groupby(['fname', 'sequence']).apply(splitStates)

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    x = jsonParser(fname, inputpath='../PyroNear/WildFire/')
    print(x.keypoints)
    print(x.states)
