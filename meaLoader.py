import os
from typing import List


def load_mea_files(directory: str = "IMS") -> List[str]:
    """
    Recursively find all .mea files in the specified directory.
    
    Args:
        directory: Root directory to search for .mea files
        
    Returns:
        List of full paths to all .mea files found
        
    Raises:
        FileNotFoundError: If the specified directory doesn't exist
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")
    
    mea_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mea"):
                mea_files.append(os.path.join(root, file))
    
    return mea_files


def main():
    """Main entry point for the MEA file finder."""
    try:
        mea_files = load_mea_files()
        print(f"Found {len(mea_files)} .mea files.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
