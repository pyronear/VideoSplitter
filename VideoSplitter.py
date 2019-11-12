from bisect import bisect_left, bisect_right
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from utils import *

class VideoSplitter:
    """
    To split a video into sequences of frames with the same camera settings (coordinates x,y,z, location name).

    The coordinates are extracted by applying OCR to the bottom of the frames.
    Since OCR is very slow, in order to avoid analysing all frames the splitting is done using binary search
    and assuming the same settings/coordinates only appear in one continuous sequence.

    Args:
        fname: video file name
        acceptCloseMatches: accept locations with similar names to try to bypass OCR problems
        max_frames: int, default: 200. Maximum number of frames to analyse
        img_preprocessing (optional): function used to prepare image for OCR
        frame_to_string (optional): function used to extract caption from frame
        extract_coordinates (optional): function used to extract coordinates from caption
    """
    def __init__(self, fname, acceptCloseMatches = True, max_frames = 200,
                 frame_to_string=frame_to_string,
                 img_preprocessing=prepareOCR,
                 extract_coordinates=extract_coordinates):
        self.fname = fname
        self.acceptCloseMatches = acceptCloseMatches
        self.max_frames = max_frames
        self.img_preprocessing = img_preprocessing
        self.frame_to_string = frame_to_string
        self.extract_coordinates = extract_coordinates

        self.video = cv2.VideoCapture(fname)
        self.Nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.captions = {} # (frame_index, caption)
        self.coordinates = {} # (frame_index, camera position in x,y,z) N.B.: strings, not float
        # (coordinates, first frame where coordinates appeared):
        # represents a sorted list of frames, used for defining the start and end of sequences by bisection
        self.seqID = {}
        self.sequences = {}

    def loadFrame(self, frame_index):
        """
        Load and return the frame with the given index or None if reading fails
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self.video.read()
        return frame if success else None

    def getCoordinates(self, frame_index):
        """
        Return the coordinates of the camera for the frame with the given index.

        If requested for the first time, store the coordinates in self.coordinates,
        the caption in self.caption and check if the frame belongs to a sequence previously identified
        or define a new entry in self.seqID
        """
        try:
            return self.coordinates[frame_index]
        except KeyError: # first time
            if len(self.coordinates) >= self.max_frames:
                raise RuntimeError(f'Maximum number of frames ({self.max_frames}) analysed')
            frame = self.loadFrame(frame_index)
            preprocessed = self.img_preprocessing(frame)
            caption = self.frame_to_string(preprocessed)
            try:
                coordinates = self.extract_coordinates(caption,
                                                       possible_matches=self.seqID if self.acceptCloseMatches else [])
            except ValueError as error:
                coordinates = ('EXTRACTION FAILED', frame_index, caption)

            self.captions[frame_index] = caption
            self.coordinates[frame_index] = coordinates
            # If first time the coordinates appeared, add a new item to seqID
            if coordinates not in self.seqID:
                self.seqID[coordinates] = frame_index
            return coordinates

    def printCaptions(self):
        "Print captions"
        print ('Frame \t Caption')
        for i in sorted(x.captions.items()): print (f'{i[0]} \t {i[1]}')

    def __getitem__(self, item):
        """
        Return the first frame that revealed the sequence to which the given frame belongs.
        This method is called by bisect to find the boundaries of the sequence
        """
        return self.seqID[ self.getCoordinates(item) ]

    def findSequences(self):
        """
        Fill dictionary self.sequences with coordinates, (first frame, last frame) for each sequence
        """
        if not self.coordinates:
            # fill coordinates with first and last values if empty
            self[0], self[self.Nframes-1]
        while True:
            # Coordinates not yet analysed
            missing = dict((frame, coord) for (frame, coord) in self.coordinates.items() \
                           if coord not in self.sequences)
            if not missing: return
            # Find the start and end of each sequence corresponding to each set of coordinates
            for (frame, coord) in missing.items():
                self.sequences[coord] = bisect_left(self, self[frame]), bisect_right(self, self[frame])

    def printSequences(self):
        """
        Print frame range and coordinates of each sequence
        """
        if not self.sequences:
            return
        print ('Frames \t Coordinates')
        for v,k in sorted((v,k) for k,v in self.sequences.items()):
            print (f'{v} \t {k}')


    def saveSequences(self, outputdir, min_frames=10):
        """
        Save each sequence to outputdir as <fname>_seq<suffix>.<ext> where suffix is 0, 1 ...

        Args:
            outputdir: str, output directory
            min_frames: int, default: 10. Minimum number of frames with same settings to consider a sequence
        """
        self.findSequences()
        import os
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
        valid_sequences = filter(lambda x: x[1] - x[0] >= min_frames, self.sequences.values())
        for i, (fmin, fmax) in enumerate(sorted(valid_sequences)):
            basename, ext = os.path.splitext(os.path.basename(self.fname))
            fname = os.path.join(outputdir, f'{basename}_seq{i}{ext}' )
            ffmpeg_extract_subclip(self.fname, fmin/self.fps, fmax/self.fps, fname)

    def __len__(self):
        return self.Nframes

import unittest
def getRef():
    """
    Return a dictionary containing the test parameters
    """
    import urllib
    url =   'https://gist.githubusercontent.com/blenzi/82746e11119cb88a67603944869e29e2/raw' # noqa: E501
    return eval(urllib.request.urlopen(url).read())


class VideoTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        "Setup only once for all tests"
        cls.ref = getRef()
        import os
        if os.path.exists(cls.ref['fname']):
            fname = cls.ref['fname']
        else:
            import pafy
            cls.vPafy = pafy.new(cls.ref['url'])
            cls.play = cls.vPafy.getbestvideo(preftype="webm")
            fname = cls.play.url
        cls.splitter = VideoSplitter(fname)

    def test_loadFrame(self):
        frame = self.splitter.loadFrame(self.ref['extract']['frame'])
        self.assertEqual(len(frame.shape), 3)

    def test_findSequences(self):
        "Test frame range in sequences (ignore exact coordinates)"
        self.maxDiff = None
        self.splitter.findSequences()
        seqs = self.splitter.sequences
        inv_seqs = dict(map(reversed, seqs.items())) # invert keys and values
        self.assertEqual(inv_seqs.keys(), self.ref['sequences'].keys())


if __name__ == '__main__':
    unittest.main()
