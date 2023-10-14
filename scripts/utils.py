import os


def make_dir_recursively_if_not_exists(path: str):
    """Make a directory recursively if it does not exist.

    Args:
        path: Path to the directory.
    """
    splits = path.split("/")
    for i in range(1, len(splits) + 1):
        dir_name = "/".join(splits[:i])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
