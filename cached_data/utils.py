import os
defaultCacheDir = 'cached_data'

def flexCachedir(subdir:str = None):
    if subdir:
        targetDir = os.path.join(defaultCacheDir, subdir)
    else:
        targetDir = defaultCacheDir
        
    if not os.path.exists(targetDir):
    # Try appending the root folder to defaultCacheDir
        root_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        other_root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
        modified_cache_dir = os.path.join(root_dir, targetDir)
        other_modified_cache_dir = os.path.join(other_root_dir, targetDir)

        if os.path.exists(other_modified_cache_dir):
            # Update defaultCacheDir to include the root folder
            return other_modified_cache_dir
        elif os.path.exists(modified_cache_dir):
            # Update defaultCacheDir to include the root folder
            return modified_cache_dir
        else:
            raise FileNotFoundError(f"Cache directory not found: {targetDir} or {modified_cache_dir}")
    else:
        return targetDir