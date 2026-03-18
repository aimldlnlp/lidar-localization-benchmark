from massive_lidar_benchmark.config.schema import ProjectConfig
from massive_lidar_benchmark.viz.animations import _frame_indices


def test_frame_stride_reduces_number_of_frames() -> None:
    config = ProjectConfig()
    config.render.frame_stride = 2
    config.render.max_frames = None

    frame_indices = _frame_indices(10, config)

    assert frame_indices == [0, 2, 4, 6, 8, 9]
