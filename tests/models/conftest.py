import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
