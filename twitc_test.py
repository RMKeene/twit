from twit import *
import numpy as np
import numpy.testing as npt
import unittest
import numpy.testing as nt
from typing import List, Tuple

import twitc

class TestTwitc(unittest.TestCase):
    def setUp(self):
        pass


    def test_AA_interp(self):
        a = twitc.generate_twit_list(0.1)
        self.assertEqual(a, 42.0)
        pass

if __name__ == '__main__':

    unittest.main(verbosity=2)
