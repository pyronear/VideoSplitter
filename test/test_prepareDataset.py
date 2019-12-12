import unittest
import pandas as pd
import numpy as np
import prepareDataset


class PrepareDatasetTester(unittest.TestCase):
    """
    Test functions in prepareDataset
    """
    def test_getSets(self):
        a = pd.DataFrame({'fBase': range(1000)})
        x = prepareDataset.getSets(a)
        self.assertEqual(x.shape, (1000, 2))

    def test_getSets2(self):
        "Test getSets with different arguments"
        a = pd.DataFrame({'a': np.repeat(range(2), 500), 'b': np.repeat(range(10), 100)})
        x = prepareDataset.getSets(a, groupby=['a', 'b'], choices=[0, 1], p=[0.3, 0.7])
        self.assertEqual(x.shape, (1000, 3))

    def test_bestSets(self):
        "Test probabilities in bestSets to 1% in inhomogeneous groups"
        p = [0.4, 0.3, 0.2, 0.1]
        choices = [0, 1, 2, 3]

        # Generate
        n = np.random.choice([2, 4, 6], size=50)
        a = pd.DataFrame({'fBase': np.repeat(range(len(n)), n)})

        # Assert
        b = prepareDataset.bestSets(a, choices=choices, p=p)
        for i, j in zip(choices, p):
            np.testing.assert_almost_equal((b.dest == i).mean(), j, decimal=2)


if __name__ == '__main__':
    unittest.main()
