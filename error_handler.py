"""Error handling and validation module for BLIP training."""

import os
import logging
import torch
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_training_config(args: Dict[str, Any]) -> bool:
    """Validate training configuration.
    
    Args:
        args (Dict[str, Any]): Training arguments
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating training configuration")
    
    # Validate model name
    if not args.get("model_name"):
        logger.error("Model name is required")
        return False
    
    # Validate data path
    data_path = args.get("data_path", "./data")
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return False
    
    # Validate batch size
    batch_size = args.get("batch_size", 16)
    if batch_size <= 0:
        logger.error("Batch size must be positive")
        return False
    
    # Validate learning rate
    learning_rate = args.get("learning_rate", 5e-5)
    if learning_rate <= 0:
        logger.error("Learning rate must be positive")
        return False
    
    # Validate epochs
    num_epochs = args.get("num_epochs", 3)
    if num_epochs <= 0:
        logger.error("Number of epochs must be positive")
        return False
    
    # Validate GPU id
    gpu_id = args.get("gpu_id", 0)
    if gpu_id < 0:
        logger.error("GPU ID must be non-negative")
        return False
    
    # Validate output directory
    output_dir = args.get("output_dir", "./output")
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output directory: {str(e)}")
        return False
    
    logger.info("Training configuration validation passed")
    return True

def validate_model(model, processor) -> bool:
    """Validate that the model and processor are loaded correctly.
    
    Args:
        model: Trained model object
        processor: Processor object
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating model and processor")
    
    if model is None:
        logger.error("Model is None")
        return False
    
    if processor is None:
        logger.error("Processor is None")
        return False
    
    try:
        # Try to access model attributes
        _ = model.config
        _ = processor
        logger.info("Model and processor validation passed")
        return True
    except Exception as e:
        logger.error(f"Model/processor validation failed: {str(e)}")
        return False

def validate_dataset(dataset) -> bool:
    """Validate dataset.
    
    Args:
        dataset: Dataset object to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating dataset")
    
    if dataset is None:
        logger.error("Dataset is None")
        return False
    
    try:
        length = len(dataset)
        if length <= 0:
            logger.warning("Dataset is empty")
            return False
        logger.info(f"Dataset validation passed, size: {length}")
        return True
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False

def validate_device(device: torch.device) -> bool:
    """Validate device configuration.
    
    Args:
        device: Device object to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating device configuration")
    
    if device is None:
        logger.error("Device is None")
        return False
    
    # For CUDA, check if GPU is available
    if device.type == "cuda":
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return False
        logger.info(f"Using CUDA device: {device}")
        return True
    
    logger.info(f"Using CPU device: {device}")
    return True

def safe_save_model(model, output_path: str, checkpoint_name: str = "checkpoint") -> bool:
    """Safely save model with error handling.
    
    Args:
        model: Model to save
        output_path (str): Output path
        checkpoint_name (str): Checkpoint name
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Create checkpoint path
        checkpoint_path = os.path.join(output_path, checkpoint_name)
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save model
        model.save_pretrained(checkpoint_path)
        logger.info(f"Model saved successfully to {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        return False

def safe_load_model(model_path: str) -> Optional[object]:
    """Safely load model with error handling.
    
    Args:
        model_path (str): Path to model
        
    Returns:
        Optional[object]: Loaded model or None if failed
    """
    try:
        # Check if path exists
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return None
        
        # Load model
        from transformers import BlipForConditionalGeneration
        model = BlipForConditionalGeneration.from_pretrained(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        return None

def validate_training_data(image_paths: list, captions: list) -> bool:
    """Validate training data.
    
    Args:
        image_paths (list): List of image paths
        captions (list): List of captions
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Validating training data")
    
    if not image_paths:
        logger.error("No image paths provided")
        return False
    
    if not captions:
        logger.error("No captions provided")
        return False
    
    if len(image_paths) != len(captions):
        logger.error("Number of image paths must match number of captions")
        return False
    
    # Check if image files exist
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            logger.warning(f"Image file {image_path} does not exist (index {i})")
    
    logger.info("Training data validation passed")
    return True

def setup_logging(log_file: str = "blip_training.log") -> None:
    """Setup logging configuration.
    
    Args:
        log_file (str): Log file name
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging setup complete, log file: {log_file}")

