#!/usr/bin/env python3
"""
Data validation script for Gemma 3 27B fine-tuning
Checks if the data file is in the correct format and validates its contents
"""

import json
import os
import argparse
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def validate_and_convert_data(input_file, output_file=None, fix_issues=False):
    """
    Validate the data file and optionally convert it to the correct format
    
    Args:
        input_file: Path to the input data file
        output_file: Path to save the converted data (if fix_issues=True)
        fix_issues: Whether to fix issues and save to output_file
        
    Returns:
        bool: True if validation passed or data was fixed, False otherwise
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist")
        return False
    
    logger.info(f"Validating data file: {input_file}")
    
    # Count lines in file first
    with open(input_file, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    # Statistics
    valid_count = 0
    invalid_count = 0
    fixed_count = 0
    formats_found = {}
    errors = []
    fixed_data = []
    
    # Read and validate each line
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=line_count, desc="Validating")):
            try:
                # Try to parse JSON
                example = json.loads(line.strip())
                
                # Track which format we found
                if "messages" in example:
                    format_type = "messages"
                elif "prompt" in example and "completion" in example:
                    format_type = "prompt-completion"
                elif "question" in example and "answer" in example:
                    format_type = "question-answer"
                elif "input" in example and "output" in example:
                    format_type = "input-output"
                elif "text" in example:
                    format_type = "text"
                else:
                    format_type = "unknown"
                    
                formats_found[format_type] = formats_found.get(format_type, 0) + 1
                
                # Check if the format is valid or can be fixed
                if format_type == "unknown":
                    invalid_count += 1
                    error_msg = f"Line {i+1}: Unknown format with keys {list(example.keys())}"
                    errors.append(error_msg)
                    
                    # Try to fix if requested
                    if fix_issues:
                        # Look for potential keys that might contain user input
                        user_content = None
                        assistant_content = None
                        
                        # Try to find user content
                        for key in ["query", "question", "prompt", "input", "user_input", "request"]:
                            if key in example and example[key]:
                                user_content = example[key]
                                break
                                
                        # Try to find assistant content
                        for key in ["response", "answer", "completion", "output", "assistant_output", "reply"]:
                            if key in example and example[key]:
                                assistant_content = example[key]
                                break
                        
                        # If we found both, create a fixed version
                        if user_content and assistant_content:
                            fixed_example = {
                                "messages": [
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": assistant_content}
                                ]
                            }
                            fixed_data.append(fixed_example)
                            fixed_count += 1
                            logger.debug(f"Fixed line {i+1}")
                else:
                    valid_count += 1
                    if fix_issues:
                        # Copy as-is for valid examples
                        fixed_data.append(example)
                        
            except json.JSONDecodeError:
                invalid_count += 1
                error_msg = f"Line {i+1}: Invalid JSON"
                errors.append(error_msg)
                logger.debug(error_msg)
    
    # Print statistics
    logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
    logger.info(f"Formats found: {formats_found}")
    
    if errors:
        logger.info(f"Found {len(errors)} errors")
        if len(errors) <= 10:
            for error in errors:
                logger.info(error)
        else:
            for error in errors[:5]:
                logger.info(error)
            logger.info("... and more errors")
    
    # Save fixed data if requested
    if fix_issues and output_file and fixed_data:
        logger.info(f"Saving fixed data to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in fixed_data:
                f.write(json.dumps(example) + '\n')
        logger.info(f"Fixed {fixed_count} examples out of {invalid_count} invalid examples")
    
    return valid_count > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and fix data for Gemma 3 fine-tuning")
    parser.add_argument("--input", type=str, default="/home/models/FAISS_INGEST/scraped_data.json", 
                        help="Input data file path")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output file path for fixed data (optional)")
    parser.add_argument("--fix", action="store_true", 
                        help="Fix issues and save to output file")
    
    args = parser.parse_args()
    
    # Run validation
    success = validate_and_convert_data(args.input, args.output, args.fix)
    
    if success:
        logger.info("Validation successful - data file can be used for fine-tuning")
        if args.fix and args.output:
            logger.info(f"Fixed data saved to {args.output}")
    else:
        logger.error("Validation failed - please check and fix the data file")
        if not args.fix and not args.output:
            logger.info("Try running with --fix and --output options to automatically fix issues") 