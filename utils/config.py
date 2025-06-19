import os
from utils.log import init_cache_db
from utils.singleton import Singleton
import json
from typing import Literal


class Config(metaclass=Singleton):
    """Class to store and load global configuration"""
    max_num_stores_loads: int
    """The maximum number of loads in the nested loops"""
    max_num_loops: int
    """The max number of nested loops"""
    max_num_load_store_dim: int
    """The max number of dimensions in load/store buffers"""
    num_tile_sizes: int
    """The number of tile sizes"""
    num_transformations: int
    """The number of transformations"""
    force_optimization: bool
    """Flag that forces the model to apply at least one optimization on every operation"""
    vect_size_limit: int
    """Vectorization size limit to prevent large sizes vectorization"""
    openmp_num_threads : int
    """the number of threads given to openmp"""
    use_bindings: bool
    """Flag to enable using python bindings for execution, if False, the execution will be done using the command line. Default is False."""
    use_vectorizer: bool
    """Flag to enable using the vectorizer C++ program for vectorization, if False, vectorization is done using transform dialect directly. Default is False."""
    data_format: Literal["json", "mlir"]
    """The format of the data, can be either "json" or "mlir". "json" mode reads json files containing benchmark features, "mlir" mode reads mlir code files directly and extract features from it using AST dumper. Default is "json"."""
    optimization_mode: Literal["last", "all"]
    """The optimization mode to use, "last" will optimize only the last operation, "all" will optimize all operations in the code. Default is "last"."""
    dataset_length : int
    """the number of instance that will be used in the training and evaluation set"""
    cache_file: str
    """The path to the cache file that will be used to store execution times (to optimize the run's runtime)"""

    empty_penalty : float
    """penalize the model if it decides to not do any optimization"""
    layer_norm_eps: float
    """epsilon value for the lstm layer norm"""
    execution_error_penalty : float
    trans_failed_penalty : float
    bias_values : list[float]
    use_lr_scheduling: bool
    residual: bool
    tree_type: Literal["ast", "raw_op"]
    
    benchmarks_folder_path: str
    """Path to the benchmarks folder. Can be empty if data format is set to "json"."""
    len_trajectory: int
    """Length of the trajectory"""
    ppo_batch_size: int
    """Batch size for PPO"""
    train_eval_split: float
    """Split ratio for train/eval set"""
    nb_iterations: int
    """Number of iterations"""
    ppo_epochs: int
    """Number of epochs for PPO"""
    entropy_coef: float
    """Entropy coefficient"""
    lr: float
    """Learning rate"""
    truncate: int
    """Maximum number of steps in the schedule"""
    json_file: str
    """Path to the JSON file containing the benchmarks code or features."""
    tags: list[str]
    """List of tags to add to the neptune experiment"""
    logging: bool
    """Flag to enable logging to neptune"""

    loaded: bool
    """Flag to check if the config was already loaded from JSON file or not"""

    def __init__(self):
        """Initialize the default values"""
        self.max_num_stores_loads = 7
        self.max_num_loops = 7
        self.max_num_load_store_dim = 7
        self.num_tile_sizes = 7
        self.num_transformations = 5
        self.force_optimization = False
        self.vect_size_limit = 512
        self.openmp_num_threads = 8
        self.use_bindings = False
        self.use_vectorizer = False
        self.data_format = "json"
        self.optimization_mode = "last"
        self.dataset_length = 0
        self.cache_file = ""
        self.benchmarks_folder_path = ""
        self.len_trajectory = 64
        self.ppo_batch_size = 64
        self.nb_iterations = 10000
        self.train_eval_split = 0
        self.ppo_epochs = 4
        self.entropy_coef = 0.01
        self.lr = 0.001
        self.truncate = 5
        self.json_file = ""
        self.tags = []
        self.logging = True
        self.loaded = False
        self.empty_penalty = 0
        self.layer_norm_eps = 0.00001
        self.execution_error_penalty = 20
        self.trans_failed_penalty = 5
        self.bias_values = [0.0, 0.3, -0.2, -0.3, 0.0, 0.0, 0.4]
        self.use_lr_scheduling = False
        self.residual = False
        self.tree_type = "ast"

    def load_from_json(self):
        """Load the configuration from the JSON file."""
        # Open the JSON file
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as f:
            config = json.load(f)
        # Set the configuration values
        self.max_num_stores_loads = config["max_num_stores_loads"]
        self.max_num_loops = config["max_num_loops"]
        self.max_num_load_store_dim = config["max_num_load_store_dim"]
        self.num_tile_sizes = config["num_tile_sizes"]
        self.num_transformations = config["num_transformations"]
        self.force_optimization = config["force_optimization"] 
        self.vect_size_limit = config["vect_size_limit"]
        self.openmp_num_threads = config["openmp_num_threads"]
        self.use_bindings = config["use_bindings"]
        self.use_vectorizer = config["use_vectorizer"]
        self.data_format = config["data_format"]
        self.optimization_mode = config["optimization_mode"]
        self.dataset_length = config['dataset_length']
        self.cache_file = config["cache_file"]
        self.benchmarks_folder_path = config["benchmarks_folder_path"]
        self.len_trajectory = config["len_trajectory"]
        self.ppo_batch_size = config["ppo_batch_size"]
        self.train_eval_split = config["train_eval_split"]
        self.nb_iterations = config["nb_iterations"]
        self.ppo_epochs = config["ppo_epochs"]
        self.entropy_coef = config["entropy_coef"]
        self.lr = config["lr"]
        self.truncate = config["truncate"]
        self.json_file = config["json_file"]
        self.tags = config["tags"]
        self.logging = config["logging"]
        
        self.empty_penalty = config["empty_penalty"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.execution_error_penalty = config["execution_error_penalty"]
        self.trans_failed_penalty = config["trans_failed_penalty"]
        self.bias_values = config["bias_values"]
        self.use_lr_scheduling = config["use_lr_scheduling"]
        self.residual = config["residual"]
        self.tree_type = config["tree_type"]

        self.num_attention_heads = config["num_attention_heads"] if "num_attention_heads" in config else 0
        
        # Check the configuration values
        assert self.train_eval_split >= 0 and self.train_eval_split <= 1, "train_eval_split should be between 0 and 1."
        assert self.data_format in ["json", "mlir"], "Invalid data format. Should be 'json' or 'mlir'."
        assert self.optimization_mode in ["last", "all"], "Invalid optimization mode. Should be 'last' or 'all'."
        # assert len(self.benchmarks_folder_path) > 0 or self.data_format == "json", "Benchmark folder path should be set if data_format is 'mlir'."
        assert self.openmp_num_threads > 0, "Openmp threads number has to be strictly positive"
        assert self.dataset_length >= 0, "the number of instances cannot be negative"
        assert len(self.bias_values) == self.num_transformations or self.bias_values == [], "Length of bias_values must match num_transformations"
        assert all(isinstance(v, float) for v in self.bias_values), "All bias_values must be float"
        assert self.tree_type in ["ast", "raw_op"]
        assert self.empty_penalty >= 0
        assert self.num_attention_heads >=0

        if self.cache_file:
            if self.cache_file.endswith(".json"):
                if not os.path.exists(self.cache_file):
                    with open(self.cache_file, "w") as g:
                        json.dump({}, g)
                else:
                    with open(self.cache_file, "r+") as g:
                        if g.read() == "":
                            json.dump({}, g)

            elif self.cache_file.endswith(".sqlite"): 
                init_cache_db(self.cache_file)

            else:
                self.cache_file = ""
            

        os.environ["OMP_NUM_THREADS"] = str(self.openmp_num_threads)

        
        # assert self.data_format != "json" or not self.use_bindings, "The specific case of using python bindings with JSON data format is not implemented yet."
        # Set loaded flag
        self.loaded = True

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            "max_num_stores_loads": self.max_num_stores_loads,
            "max_num_loops": self.max_num_loops,
            "max_num_load_store_dim": self.max_num_load_store_dim,
            "num_tile_sizes": self.num_tile_sizes,
            "num_transformations": self.num_transformations,
            "force_optimization": self.force_optimization,
            "vect_size_limit": self.vect_size_limit,
            "openmp_num_threads": self.openmp_num_threads,
            "use_bindings": self.use_bindings,
            "use_vectorizer": self.use_vectorizer,
            "data_format": self.data_format,
            "optimization_mode": self.optimization_mode,
            "dataset_length":self.dataset_length,
            "cache_file": self.cache_file,
            "benchmarks_folder_path": self.benchmarks_folder_path,
            "len_trajectory": self.len_trajectory,
            "ppo_batch_size": self.ppo_batch_size,
            "train_eval_split": self.train_eval_split,
            "nb_iterations": self.nb_iterations,
            "ppo_epochs": self.ppo_epochs,
            "entropy_coef": self.entropy_coef,
            "lr": self.lr,
            "truncate": self.truncate,
            "json_file": self.json_file,
            "tags": self.tags,
            "logging": self.logging
        }

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.__dict__)
