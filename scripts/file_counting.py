import os
import pathlib

def count_folders_and_files(directory_path, recursive=False):
    """
    Count the number of subfolders and files in a given directory.
    
    Args:
        directory_path (str): Path to the directory to analyze
        recursive (bool, optional): If True, count files and folders in all subdirectories. 
                                    If False, only count in the immediate directory. 
                                    Defaults to False.
    
    Returns:
        dict: A dictionary containing counts of folders and files
    """
    # Normalize the path and expand user directory if needed
    directory = pathlib.Path(directory_path).expanduser().resolve()
    
    # Validate directory exists
    if not directory.is_dir():
        raise ValueError(f"The provided path is not a valid directory: {directory}")
    
    # Initialize counters
    folder_count = 0
    file_count = 0
    
    if not recursive:
        # Count only in the immediate directory
        for item in directory.iterdir():
            if item.is_dir():
                folder_count += 1
            elif item.is_file():
                file_count += 1
    else:
        # Recursive counting using os.walk
        for root, dirs, files in os.walk(directory):
            folder_count += len(dirs)
            file_count += len(files)
    
    return {
        "total_folders": folder_count,
        "total_files": file_count,
        "directory": str(directory)
    }

def print_folder_file_summary(directory_path, recursive=False):
    """
    Print a formatted summary of folder and file counts.
    
    Args:
        directory_path (str): Path to the directory to analyze
        recursive (bool, optional): If True, count files and folders in all subdirectories. 
                                    Defaults to False.
    """
    try:
        summary = count_folders_and_files(directory_path, recursive)
        print(f"Directory Analyzed: {summary['directory']}")
        print(f"Total Folders: {summary['total_folders']}")
        print(f"Total Files: {summary['total_files']}")
        print(f"Total Items: {summary['total_folders'] + summary['total_files']}")
        print(f"Counting Mode: {'Recursive' if recursive else 'Non-Recursive'}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    directory_to_analyze = '/home/shaunliew/ai-laryngeal-video-based-classifier/iqm_filtered_dataset'
    
    # Non-recursive count (only immediate directory)
    print("Non-Recursive Count:")
    print_folder_file_summary(directory_to_analyze, recursive=False)
    
    print("\n" + "-"*40 + "\n")
    
    # Recursive count (including all subdirectories)
    print("Recursive Count:")
    print_folder_file_summary(directory_to_analyze, recursive=True)