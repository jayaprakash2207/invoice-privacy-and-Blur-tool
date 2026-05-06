def test_imports():
    import invoice_masker.main
    import invoice_masker.api
    import invoice_masker.utils.config


def test_config_loads():
    from invoice_masker.utils.config import load_config

    cfg = load_config()
    assert cfg is not None
