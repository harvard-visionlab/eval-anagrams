

def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs network access (HF Hub download)")
