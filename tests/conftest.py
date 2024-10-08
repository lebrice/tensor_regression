import pytest
import torch


@pytest.fixture
def device(request: pytest.FixtureRequest) -> torch.device:
    device = getattr(request, "param", None)
    if device:
        assert isinstance(device, torch.device | str)
        # a device was specified with indirect parametrization.
        return torch.device(device) if isinstance(device, str) else device
    if torch.cuda.is_available():
        return torch.device("cuda", index=torch.cuda.current_device())
    return torch.device("cpu")
