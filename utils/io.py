import pickle


def load_pickle_file(file_name: str):
    """
    Load a pickle file.

    :param file_name: str.
    :return: an object stored in the pickle file.
    """
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_to_pickle_file(obj, file_name: str) -> None:
    """
    Save an object in a pickle file.

    :param obj: an object to be saved.
    :param file_name: str.
    :return: None
    """
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)
