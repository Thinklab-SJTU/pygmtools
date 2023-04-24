import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="",
        choices=["numpy", "paddle", "jittor", "tensorflow", "mindspore"],
        help="The backend that need to be tested in classic solvers."
    )


@pytest.fixture(scope='session')
def get_backend(request):
    return request.config.getoption("--backend")
