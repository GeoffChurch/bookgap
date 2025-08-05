#!/usr/bin/env python3
"""
Script to check the status of OpenAI batch jobs and download results if completed.
"""

import os
from pathlib import Path
from .batch_operations import (
    list_batch_jobs, 
    get_batch_status, 
    process_completed_batch,
    monitor_batch_job
)

def interactive_menu():
    """Interactive menu for batch operations."""
    print("OpenAI Batch Job Status Checker")
    print("================================")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        return
    
    while True:
        print("\nOptions:")
        print("1. List all batch jobs")
        print("2. Check specific batch status")
        print("3. Download and extract results for completed batch")
        print("4. Monitor batch job until completion")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            list_batch_jobs()
        elif choice == "2":
            batch_id = input("Enter batch ID: ").strip()
            if batch_id:
                get_batch_status(batch_id)
        elif choice == "3":
            batch_id = input("Enter batch ID: ").strip()
            if batch_id:
                stats = process_completed_batch(batch_id)
                if stats:
                    print(f"Successfully processed batch {batch_id}")
        elif choice == "4":
            batch_id = input("Enter batch ID: ").strip()
            if batch_id:
                print("Starting monitoring... (Ctrl+C to stop)")
                try:
                    batch_status = monitor_batch_job(batch_id)
                    if batch_status and batch_status.status == "completed":
                        # print("\nBatch completed! Downloading and extracting results...")
                        # stats = process_completed_batch(batch_id)
                        # if stats:
                        #     print(f"Successfully processed batch {batch_id}")
                        input("\nBatch completed! Download and extract results? (y/N): ")
                        if input().strip().lower() == 'y':
                            stats = process_completed_batch(batch_id)
                            if stats:
                                print(f"Successfully processed batch {batch_id}")
                except KeyboardInterrupt:
                    print("\nMonitoring stopped by user.")
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main function."""
    interactive_menu()

if __name__ == "__main__":
    main()
