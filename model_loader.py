"""Model loading module for BLIP training."""

import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import logging

logger = logging.getLogger(__name__)

def load_blip_model(model_name, model_type):
    """Load BLIP model for training.
    
    Args:
        model_name (str): Name or path of the model to load
        model_type (str): Type of BLIP model (blip or blip2)
        
    Returns:
        model: Loaded model
        processor: Corresponding processor
    """
    logger.info(f"Loading {model_type} model: {model_name}")
    
    try:
        if model_type == blip:
            # Load BLIP model
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            processor = BlipProcessor.from_pretrained(model_name)
            
        elif model_type == blip2:
            # Load BLIP2 model
            model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            processor = Blip2Processor.from_pretrained(model_name)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Check if GPU is available and move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model.to(device)
        logger.info("Model loaded successfully")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def load_clip_model(model_name):
    """Load CLIP model for image feature extraction.
    
    Args:
        model_name (str): Name or path of the CLIP model to load
        
    Returns:
        model: Loaded CLIP model
        processor: Corresponding processor
    """
    logger.info(f"Loading CLIP model: {model_name}")
    
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # Check if GPU is available and move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model.to(device)
        logger.info("CLIP model loaded successfully")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {str(e)}")
        raise
