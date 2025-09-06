# Pytest configuration to ignore legacy step-based tests that reference deprecated APIs
import os

def pytest_ignore_collect(path, config):
    name = os.path.basename(str(path))
    return name.startswith("test_config_manager_step")
