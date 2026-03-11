import argparse

def parse_arguments():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="BLIP Training Script")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base",
                        help="Name of the BLIP model to use (default: Salesforce/blip-image-captioning-base)")
    parser.add_argument("--model_type", type=str, choices=["blip", "blip2"], default="blip",
                        help="Type of BLIP model to use (default: blip)")
    
    # Data configuration
    parser.add_argument("--dataset_name", type=str, default="coco",
                        help="Name of the dataset to use (default: coco)")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset (default: ./data)")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (default: 1)")
    
    # Hardware configuration
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for training (default: 0)")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--gradient_clipping", action="store_true",
                        help="Enable gradient clipping")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model checkpoints (default: ./output)")
    parser.add_argument("--logging_dir", type=str, default="./logs",
                        help="Directory to save training logs (default: ./logs)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps (default: 500)")
    
    # PEFT configuration
    parser.add_argument("--use_peft", action="store_true",
                        help="Enable parameter-efficient fine-tuning")
    parser.add_argument("--peft_method", type=str, choices=["lora", "adapter"], default="lora",
                        help="PEFT method to use (default: lora)")
    
    return parser.parse_args()
