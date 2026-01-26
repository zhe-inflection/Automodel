#!/usr/bin/env python3
"""
Convert chat JSONL data to simple instruction format (question/answer).
This avoids chat template formatting during training which can cause hanging.
"""

import json
import argparse


def convert_chat_to_instruction(input_file, output_file):
    """Convert chat JSONL to simple instruction format."""
    
    converted_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                if not line.strip():
                    continue
                    
                data = json.loads(line.strip())
                messages = data.get("messages")
                if not messages:
                    skipped_count += 1
                    continue
                
                # Extract the last user message as question and last assistant message as answer
                question = ""
                answer = ""
                
                # Find the last assistant message and its preceding user message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "assistant" and messages[i].get("content"):
                        answer = messages[i]["content"]
                        # Look for the preceding user message
                        for j in range(i - 1, -1, -1):
                            if messages[j]["role"] == "user" and messages[j].get("content"):
                                question = messages[j]["content"]
                                break
                        break
                
                if not question or not answer:
                    skipped_count += 1
                    continue
                
                # Save as simple instruction format
                output_data = {
                    "question": question,
                    "answer": answer,
                    "prompt_id": data.get("prompt_id", f"sample_{line_num}")
                }
                
                outfile.write(json.dumps(output_data) + '\n')
                converted_count += 1
                
                if converted_count % 100 == 0:
                    print(f"Processed {converted_count} samples...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1
    
    print(f"\nConversion complete:")
    print(f"  - Converted: {converted_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert chat JSONL to instruction format")
    parser.add_argument("input_file", help="Path to input chat JSONL file")
    parser.add_argument("output_file", help="Path to output instruction JSONL file")
    args = parser.parse_args()
    
    convert_chat_to_instruction(args.input_file, args.output_file)
