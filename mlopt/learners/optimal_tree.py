from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST, OPTIMAL_TREE
from mlopt.utils import pandas2array
import shutil
from subprocess import call
from warnings import warn
import os


class OptimalTree(Learner):

    def __init__(self,
                 **options):
        """
        Initialize OptimalTrees class.

        Parameters
        ----------
        options : dict
            Learner options as a dictionary.
        """
        # Define name
        self.name = OPTIMAL_TREE

        # Load Julia
        import julia
        self.jl = julia.Julia()
        # Add crypto library to path to check OptimalTrees license
        self.jl.eval("push!(Base.DL_LOAD_PATH, " +
                     "joinpath(dirname(Base.find_package(\"MbedTLS\")), " +
                     "\"../deps/usr\", Sys.iswindows() ? \"bin\" : \"lib\"))")
        # Reset random seed for repeatability
        self.jl.eval("using Random; Random.seed!(1)")
        # Define functions needed
        self._array = self.jl.eval("Array")
        self._convert = self.jl.eval("convert")
        self._create_classifier = \
            self.jl.eval("OptimalTrees.OptimalTreeClassifier")
        self._fit = self.jl.eval("OptimalTrees.fit!")
        self._predict = self.jl.eval("OptimalTrees.predict_proba")
        self._write = self.jl.eval("OptimalTrees.writejson")
        self._writedot = self.jl.eval("OptimalTrees.writedot")
        self._read = self.jl.eval("OptimalTrees.readjson")
        # NB _open function defined separately
        # to preserve consistency
        self._close = self.jl.eval("close")

        # Assign settings
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')
        self.options = {}
        self.options['hyperplanes'] = options.pop('hyperplanes', False)
        self.options['cp'] = options.pop('cp', None)
        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', N_BEST),
                                     self.n_classes)
        self.options['save_pdf'] = options.pop('save_pdf', False)
        self.optimaltrees_options = {'max_depth': 10}
        if self.options['hyperplanes']:
            self.optimaltrees_options['hyperplane_config'] = \
                self.jl.eval('(sparsity=:all,)')
        if self.options['cp']:
            self.optimaltrees_options['cp'] = self.options['cp']

    def _open(self, file_name, option):
        """
        Define this function separately to keep consistency
        of IOBuffer julia type.
        """
        return self.jl.eval("PyCall.pyjlwrap_new(open(\"%s\", \"%s\"))"
                            % (file_name, option))

    def train(self, X, y):

        # Convert X to array
        self.n_train = len(X)
        X = pandas2array(X)

        # Create classifier
        self._lnr = self._create_classifier(**self.optimaltrees_options)

        # Train classifier
        self._fit(self._lnr, X, y)

    def predict(self, X):

        # Unroll pandas dataframes
        X = pandas2array(X)

        # Evaluate probabilities
        # NB. They are returned as a DataFrame of DataFrames.jl
        #     and we convert them to an array which in python
        #     becomes a numpy array
        proba = self._predict(self._lnr, X)
        y = self._convert(self._array, proba)

        return self.pick_best_probabilities(y)

    def save(self, file_name):
        # Save tree as json file
        io = self._open(file_name + ".json", "w")
        self._write(io, self._lnr)
        self._close(io)

        # Save tree to dot file and convert it to
        # pdf for visualization purposes
        if self.options['save_pdf']:
            if shutil.which("dot") is not None:
                self._writedot(file_name + ".dot", self._lnr)
                call(["dot", "-Tpdf", "-o",
                      file_name + ".pdf",
                      file_name + ".dot"])
            else:
                warn("dot command not found in path")

    def load(self, file_name):
        # Check if file name exists
        if not os.path.isfile(file_name + ".json"):
            raise ValueError("Optimal Tree json file does not exist.")

        # Load tree from file
        io = self._open(file_name + ".json", "r")
        self._lnr = self._read(io)
        self._close(io)
