import unittest
import numpy as np
import numpy.testing as npt
import mlopt
import os

script_dir = os.path.dirname(__file__)


#  class TestReadMat(unittest.TestCase):
#      def test_read(self):
#          file_name = "afiro"
#          data = mlopt.utils.read_mat(os.path.join(script_dir, "data", file_name))
#          c_test = np.array([0, -0.4000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3200, 0, 0, 0, -0.6000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4800, 0, 0, 10.0000])
#
#          npt.assert_almost_equal(data['c'], c_test)



if __name__ == '__main__':
    unittest.main()
