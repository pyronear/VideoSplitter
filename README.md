# VideoSplitter

To split a video into sequences of frames with the same camera settings: coordinates x,y,z (zoom), location name

## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)
- [tesseract](https://github.com/tesseract-ocr/tesseract)

### Installation

`pip install opencv-python pillow pytesseract moviepy`

## Usage

`python VideoSplitter.py -h` for help

Write to the output directory:

- split videos (`<name>_seq<first_frame>_<last_frame>.<ext>`)
- info (captions, sequences, ...) in pickle files

### Known issues

- Last split video can last longer than it should. Might be related to `ffmpeg_extract_subclip`. Can be circunvented by skipping frames beyond `<last_frame> - <first_frame>` stored in filename
- Extracting the caption is not always easy...

