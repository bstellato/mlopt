from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST, FRAC_TRAIN, OPTIMAL_TREE
from mlopt.utils import pandas2array, get_n_processes
import shutil
from subprocess import call
import time
import os
import sys
import logging


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

        # Assign settings
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')
        self.options = {}
        self.options['hyperplanes'] = options.pop('hyperplanes', False)
        #  self.options['fast_num_support_restarts'] = \
        #      options.pop('fast_num_support_restarts', [20])
        self.options['parallel'] = options.pop('parallel_trees', True)
        self.options['cp'] = options.pop('cp', None)
        self.options['max_depth'] = options.pop('max_depth', [5, 10, 15])
        self.options['minbucket'] = options.pop('minbucket', [1, 5, 10])
        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', N_BEST),
                                     self.n_classes)
        self.options['save_svg'] = options.pop('save_svg', False)

        # Get fraction between training and validation
        self.options['frac_train'] = options.pop('frac_train', FRAC_TRAIN)

        # Load Julia
        import julia
        self.jl = julia.Julia()
        n_cpus = get_n_processes()

        n_cur_procs = self.jl.eval("using Distributed; nprocs()")
        if n_cur_procs < n_cpus and self.options['parallel']:
            # Add processors to match number of cpus
            self.jl.eval("addprocs(%d)" % (n_cpus - n_cur_procs))

        # Add crypto library to path to check OptimalTrees license
        path_string = "push!(Base.DL_LOAD_PATH, " + \
                      "joinpath(dirname(Base.find_package(\"MbedTLS\")), " + \
                      "\"../deps/usr\", Sys.iswindows() ? \"bin\" : \"lib\"))"
        if n_cpus > 1 and sys.platform == 'darwin' and \
                self.options['parallel']:
            # Add @everywhere if we are on a multiprocess machine
            # It seems necessary only on OSX
            path_string = "@everywhere " + path_string
        self.jl.eval(path_string)
        # Reset random seed for repeatability
        self.jl.eval("using Random; Random.seed!(1)")
        # Define functions needed
        self._array = self.jl.eval("Array")
        self._convert = self.jl.eval("convert")
        self._create_classifier = \
            self.jl.eval("OptimalTrees.OptimalTreeClassifier")
        self._create_grid = \
            self.jl.eval("OptimalTrees.GridSearch")
        self._fit = self.jl.eval("OptimalTrees.fit!")
        self._predict = self.jl.eval("OptimalTrees.predict_proba")
        self._write = self.jl.eval("OptimalTrees.writejson")
        self._writedot = self.jl.eval("OptimalTrees.writedot")
        self._read = self.jl.eval("OptimalTrees.readjson")
        # NB _open function defined separately
        # to preserve consistency
        self._close = self.jl.eval("close")

        # Assign optimaltrees options
        self.optimaltrees_options = {'ls_random_seed': 1}
        self.optimaltrees_options['max_depth'] = self.options['max_depth']
        self.optimaltrees_options['minbucket'] = self.options['minbucket']
        if self.options['hyperplanes']:
            self.optimaltrees_options['hyperplane_config'] = \
                self.jl.eval('[[(sparsity=:all,)]]')
            # Sparse hyperplanes
            #  self.optimaltrees_options['hyperplane_config'] = \
            #      self.jl.eval('[[(sparsity=2,)]]')
            #  self.optimaltrees_options['fast_num_support_restarts'] = \
            #      self.options['fast_num_support_restarts']
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

        info_str = "Training trees "
        if self.options['parallel']:
            info_str += "on %d processors" % self.jl.eval("nprocs()")
        else:
            info_str += "\n"
        logging.info(info_str)

        # Start time
        start_time = time.time()

        # Reset random seed
        self.jl.eval("using Random; Random.seed!(1)")

        # Create classifier
        # Set seed to 1 to make the validation reproducible
        self._lnr = \
            self._create_classifier(
                ls_random_seed=self.optimaltrees_options['ls_random_seed']
            )

        # Create grid search
        self._grid = self._create_grid(self._lnr,
                                       **self.optimaltrees_options)

        # Train classifier
        self._fit(self._grid, X, y,
                  train_proportion=self.options['frac_train'])

        # End time
        end_time = time.time()
        logging.info("Tree training time %.2f" % (end_time - start_time))

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
        if self.options['save_svg']:
            if shutil.which("dot") is not None:
                self._writedot(file_name + ".dot", self._lnr)
                call(["dot", "-Tsvg", "-o",
                      file_name + ".svg",
                      file_name + ".dot"])
            else:
                logging.warning("dot command not found in path")

    def load(self, file_name):
        # Check if file name exists
        if not os.path.isfile(file_name + ".json"):
            err = "Optimal Tree json file does not exist."
            logging.error(err)
            raise ValueError(err)

        # Load tree from file
        io = self._open(file_name + ".json", "r")
        self._lnr = self._read(io)
        self._close(io)
