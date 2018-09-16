import unittest
import mlo


class TestReadMPS(unittest.TestCase):
    def test_read(self):
        file_name = "afiro.mps"
        data = mlo.read_mps(file_name)

