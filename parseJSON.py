import json, numpy as np, pandas as pd

def getFps(fname):
    "Return the number of frames per second for the given movie file"
    import cv2
    return cv2.VideoCapture(f'../PyroNear/WildFire/{fname}').get(cv2.CAP_PROP_FPS)

def getFileInfo(fname, pattern = r'(?P<fBase>\w+)_seq(?P<splitStart>\d+)_(?P<splitEnd>\d+).(?P<ext>\w+)'):
    """
    Return a DataFrame with file info from the given series with fname:
    - fBase (fname without _seqX_Y),
    - splitStart and splitEnd (first and last frame for split file)
    - fps (frames per second)

    Args:
    - fname: Series
    - pattern: fname pattern to extract fBase, splitStart, splitEnd
    """
    d = fname.str.extract(pattern).astype({'splitStart': float, 'splitEnd': float}) # to allow NaN
    d['fBase'] = (d.fBase + '.' + d.ext).fillna(fname)
    d['fps'] = d.fBase.apply(getFps)
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
    new = df.rename(columns={'frame': 'stateStart'}).join(Next.frame.rename('stateEnd') - 1)
    idxLast = new.iloc[-1].name
    if not sameState(new.iloc[-1], Prev.iloc[-1]):
        new.loc[new.index[-1], 'stateEnd'] = new.splitEnd.iloc[-1]
    else:
        new.loc[new.index[-2], 'stateEnd'] += 1
    return new.dropna(subset=['stateEnd'])

class jsonParser:
    ""
    def __init__(self, fname):
        with open(fname) as jsonFile:
            info = json.load(jsonFile)

        # Annotation labels
        self.labels = pd.DataFrame(info['attribute'].values())
        self.labels['class'] = info['attribute'].keys()

        # Annotations: process and cleanup
        # - merge with 'files' to get fname, fBase, fps, splitStart, splitEnd
        # - add information from all other DataFrames (in DFS)
        # - drop columns which are not needed after processing
        # - drop lines where both fire and exploitable are NaN (no annotation)
        # - replace exploitable=NaN by True
        # - sort by fname and t
        # - drop non-exploitable
        self.annotations = pd.DataFrame(info['metadata'].values()).drop(columns='flg')

        # DataFrame with fid and fname. Add fBase, fps, splitStart, splitEnd
        # to the ones for which there are annotations
        self.files = pd.DataFrame(info['file'].values())[['fid', 'fname']]
        fnames = self.files.loc[self.files.fid.isin(self.annotations.vid), 'fname']
        self.files = self.files.join( getFileInfo(fnames) )

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



def parseJSON(fname):
    ""
    with open(fname) as jsonFile:
        info = json.load(jsonFile)

    # Movie files.
    # For split files: extract beginning and end of split and set fBase as original file
    # For non-split files: fBase = fname
    files = pd.DataFrame(info['file'].values())
    pattern = '(?P<fBase>\w+)_seq(?P<splitStart>\d+)_(?P<splitEnd>\d+).(?P<ext>\w+)'
    d = files.fname.str.extract(pattern)
    d['fBase'] = d.fBase + '.' + d.ext
    d = d.fillna({'fBase': files['fname']})
    d['fps'] = d.fBase.apply(fps)
    files = files.join(d).drop(columns=['type', 'loc', 'src', 'ext'])

    # Annotation labels
    labels = pd.DataFrame(info['attribute'].values())
    labels['class'] = info['attribute'].keys()

    # Annotations: process and cleanup
    # - merge with 'files' to get fname, fBase, fps, splitStart, splitEnd
    # - add information from all other DataFrames (in DFS)
    # - drop columns which are not needed after processing
    # - drop lines where both fire and exploitable are NaN (no annotation)
    # - replace exploitable=NaN by True
    # - sort by fname and t
    # - drop non-exploitable
    annotations = pd.DataFrame(info['metadata'].values()).drop(columns='flg')
    d = pd.merge(annotations, files, left_on='vid', right_on='fid')

    def splitKeypointValues(x):
        "Convert annotation info to a Series with the keys and values"
        return pd.Series(x).rename(index=dict(zip(labels['class'], labels['aname'])))

    DFS = [
           d['av'].apply(splitKeypointValues), # annotation info
           pd.DataFrame(d.xy.tolist(), columns=['dummy', 'x', 'y']), # split xy
           pd.DataFrame(d.z.tolist(), columns=['t']), # time of keypoint
          ]
    d = d.join(DFS).drop(columns=['fid', 'xy', 'av', 'dummy', 'vid', 'z', 'spatial'])\
         .dropna(how='all', subset=['fire', 'exploitable'])\
         .fillna({'exploitable': True}).sort_values(['fname', 't'])
    d['frame'] = np.round(d.fps * d.t) + d.splitStart
    return d.loc[d.exploitable != '0']

def getStates(df):
    return df.groupby(['fname', 'sequence']).apply(splitStates)
    #.reset_index().drop(columns=['level_2'])

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    df = parseJSON(fname)
    print(df)
    print(getStates(df))
