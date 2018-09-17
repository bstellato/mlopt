import unittest
import mlo
import os


class TestReadMPS(unittest.TestCase):
    def test_read(self):
        file_name = "afiro.mps"
        data = mlo.read_mps(os.path.join("data", file_name))



