__doc__ = "Functions to prepare Wildfire dataset for training: split images into sets like train, valid and test"

import pandas as pd
import numpy as np
from parseAnnotations import getFrameLabels, extractFrames


def getSets(labels, choices=['train', 'valid', 'test'],
            p=[0.7, 0.2, 0.1], groupby='fBase'):
    """
    Assign each row of the given DataFrame to sets like 'training', 'valid', 'test'.
    The set is chosen randomly for each group (e.g. each movie) according to the given
    probabilities.

    Args:
    - labels: DataFrame containing frame labels (see parseAnnotations.getFrameLabels)
    - choices: list, default: ['train', 'valid', 'test']
    - p: list, default: [0.7, 0.2, 0.1]. Probability of each set passed to
      numpy.random.choice
    - groupby: str or list of columns in DataFrame labels, default: 'fBase'.
      Used to define the groups

    Return:
    - DataFrame with extra column 'dest', with values like 'train', 'valid', 'test'
    """
    assert len(choices) == len(p), f'choices and probabilities (p) must have the same length, got {choices}, {p}'

    def fcn(x):
        return np.random.choice(choices, p=p)

    dest = labels.groupby(groupby).apply(fcn).rename('dest')
    return pd.merge(dest, labels, left_index=True, right_on=groupby)


def bestSets(labels, choices=['train', 'valid', 'test'],
             p=[0.7, 0.2, 0.1], groupby='fBase', max_attempts=100, tolerance=0.01):
    """
    Call getSets until the number of elements in each set matches the
    desired probabilities within the given tolerance.

    After max_attempts, returns the choice that minimises the sum of the differences
    between actual and desired probabilities

    N.B.: the number of elements per group may vary substantially so depending on the
    assignment of each group into e.g. 'training', 'valid' and 'test' sets, the sets may
    be populated with different probabilities compared to the groups.

    Args:
    - labels: DataFrame containing frame labels (see parseAnnotations.getFrameLabels)
    - choices: list, default: ['train', 'valid', 'test']
    - p: list, default: [0.7, 0.2, 0.1]. Probability of each set passed to
      numpy.random.choice
    - groupby: str or list of columns in DataFrame labels, default: 'fBase'.
      Used to define the groups
    - max_attempts: int, default: 100. Maximum calls to getSets
    - tolerance: float or list, default: 0.01. Maximum difference between the actual
      probability and the desired one to stop the iteration before max_attempts

    Returns:
    - DataFrame with extra column 'dest', with values 'train', 'valid' or 'test'
    """
    def distance(dest):
        "Distance between the desired and actual probabilities for each set"
        return abs(dest.value_counts(normalize=True).reindex(choices).fillna(0) - p)

    def loop():
        """
        Yield the result of getSets and stop if the probabilities match to
        better than the desired tolerance
        """
        for _ in range(max_attempts):
            new_labels = getSets(labels, choices, p, groupby)
            yield new_labels
            if all(distance(new_labels.dest) < tolerance):
                return

    return min(loop(), key=lambda x: distance(x.dest).sum())
