class TrainingConfig:
    def __init__(self):
        # Model architecture
        self.latent_dim = 10
        self.initial_filters = 32
        self.patch_size = 17
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 128
        self.num_epochs = 200
        self.patches_per_image = 3000
        
        # Loss weights
        self.noise_weight = 0.6
        self.center_weight = 0.4
        
        # Optimizer settings
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 1e-5
        
        # Learning rate scheduling
        self.use_lr_scheduler = True
        self.lr_step_size = 30
        self.lr_gamma = 0.5
        
        # Early stopping
        self.patience = 20
        self.min_delta = 1e-4

class ModelConfig:
    def __init__(self):
        # Encoder architecture
        self.encoder_channels = [1, 16, 32, 64]
        self.encoder_kernel_sizes = [3, 3, 3]
        self.pool_sizes = [2, 2, 2]
        
        # Decoder architectures
        self.decoder_channels = [64, 32, 16, 1]
        self.decoder_kernel_sizes = [3, 3, 3, 3]
        
        # Activation functions
        self.use_leaky_relu = True
        self.leaky_slope = 0.2
        
        # Normalization
        self.use_batch_norm = True
        self.use_instance_norm = False
        
        # Dropout
        self.use_dropout = True
        self.dropout_rate = 0.1

class ExperimentConfig:
    def __init__(self):
        self.training = TrainingConfig()
        self.model = ModelConfig()
        
        # Experiment tracking
        self.exp_name = "dual_decoder_v1"
        self.save_frequency = 10
        self.eval_frequency = 5
        
        # Data paths
        self.data_root = "../Data"
        self.save_dir = "models/weights"
        self.log_dir = "runs/dual_decoder"
        
        # Evaluation settings
        self.eval_batch_size = 64
        self.num_visualization_samples = 8
        
    def get_experiment_name(self, lattice_type):
        """Generate unique experiment name"""
        return f"{self.exp_name}_{lattice_type}_lr{self.training.learning_rate}"

# Create default configurations
default_config = ExperimentConfig()

def get_config_variant(variant_name):
    """Get specific configuration variants"""
    config = ExperimentConfig()
    
    if variant_name == "high_capacity":
        config.model.encoder_channels = [1, 32, 64, 128]
        config.model.decoder_channels = [128, 64, 32, 1]
        config.training.batch_size = 64
        
    elif variant_name == "fast_training":
        config.training.learning_rate = 0.002
        config.training.num_epochs = 100
        config.training.batch_size = 256
        
    elif variant_name == "robust":
        config.model.use_dropout = True
        config.model.dropout_rate = 0.2
        config.training.weight_decay = 1e-4
        
    return config
