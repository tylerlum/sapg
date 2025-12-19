from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class ObjectSizeDistribution:
    type: Literal[
        "hammer",
        "knife",
        "screwdriver",
        "marker",
        "spatula",
        "whiteboard_eraser",
        "phone",
    ]
    handle_min_lengths: Union[Tuple[float, float, float], Tuple[float, float]]
    handle_max_lengths: Union[Tuple[float, float, float], Tuple[float, float]]
    head_min_lengths: Optional[Union[Tuple[float, float, float], Tuple[float, float]]]
    head_max_lengths: Optional[Union[Tuple[float, float, float], Tuple[float, float]]]

    def __post_init__(self):
        assert len(self.handle_min_lengths) == len(self.handle_max_lengths), (
            f"handle_min_lengths and handle_max_lengths must have the same length: {self.handle_min_lengths} and {self.handle_max_lengths}"
        )
        assert (self.head_min_lengths is None and self.head_max_lengths is None) or (
            len(self.head_min_lengths) == len(self.head_max_lengths)
        ), (
            f"head_min_lengths and head_max_lengths must have the same length: {self.head_min_lengths} and {self.head_max_lengths}"
        )

    @property
    def shape(self) -> Literal["cuboid", "cylinder"]:
        if len(self.handle_min_lengths) == 3:
            return "cuboid"
        elif len(self.handle_min_lengths) == 2:
            return "cylinder"
        else:
            raise ValueError(f"Invalid handle min lengths: {self.handle_min_lengths}")


OBJECT_SIZE_DISTRIBUTIONS: List[ObjectSizeDistribution] = [
    ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.02, 0.02),
        handle_max_lengths=(0.3, 0.05, 0.03),
        head_min_lengths=(0.02, 0.05, 0.02),
        head_max_lengths=(0.06, 0.12, 0.06),
    ),
    ObjectSizeDistribution(
        type="hammer",
        handle_min_lengths=(0.15, 0.02),
        handle_max_lengths=(0.3, 0.04),
        head_min_lengths=(0.05, 0.02),
        head_max_lengths=(0.12, 0.06),
    ),
    ObjectSizeDistribution(
        type="knife",
        handle_min_lengths=(0.1, 0.02, 0.02),
        handle_max_lengths=(0.15, 0.05, 0.03),
        head_min_lengths=(0.08, 0.02, 0.01),
        head_max_lengths=(0.2, 0.08, 0.015),
    ),
    ObjectSizeDistribution(
        type="knife",
        handle_min_lengths=(0.1, 0.02),
        handle_max_lengths=(0.15, 0.04),
        head_min_lengths=(0.08, 0.02, 0.01),
        head_max_lengths=(0.2, 0.08, 0.015),
    ),
    ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.1, 0.02, 0.02),
        handle_max_lengths=(0.15, 0.05, 0.03),
        head_min_lengths=(0.1, 0.01, 0.01),
        head_max_lengths=(0.2, 0.015, 0.015),
    ),
    ObjectSizeDistribution(
        type="screwdriver",
        handle_min_lengths=(0.1, 0.02),
        handle_max_lengths=(0.15, 0.04),
        head_min_lengths=(0.1, 0.01, 0.01),
        head_max_lengths=(0.2, 0.015, 0.015),
    ),
    ObjectSizeDistribution(
        type="marker",
        handle_min_lengths=(0.1, 0.015, 0.015),
        handle_max_lengths=(0.15, 0.025, 0.025),
        head_min_lengths=None,
        head_max_lengths=None,
    ),
    ObjectSizeDistribution(
        type="marker",
        handle_min_lengths=(0.1, 0.015),
        handle_max_lengths=(0.15, 0.025),
        head_min_lengths=None,
        head_max_lengths=None,
    ),
    ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.12, 0.015, 0.015),
        handle_max_lengths=(0.25, 0.03, 0.03),
        head_min_lengths=(0.08, 0.06, 0.01),
        head_max_lengths=(0.12, 0.08, 0.03),
    ),
    ObjectSizeDistribution(
        type="spatula",
        handle_min_lengths=(0.12, 0.015),
        handle_max_lengths=(0.25, 0.025),
        head_min_lengths=(0.08, 0.06, 0.01),
        head_max_lengths=(0.12, 0.08, 0.03),
    ),
    ObjectSizeDistribution(
        type="whiteboard_eraser",
        handle_min_lengths=(0.1, 0.04, 0.02),
        handle_max_lengths=(0.15, 0.06, 0.03),
        head_min_lengths=None,
        head_max_lengths=None,
    ),
    ObjectSizeDistribution(
        type="whiteboard_eraser",
        handle_min_lengths=(0.1, 0.04),
        handle_max_lengths=(0.15, 0.06),
        head_min_lengths=None,
        head_max_lengths=None,
    ),
    ObjectSizeDistribution(
        type="phone",
        handle_min_lengths=(0.12, 0.06, 0.01),
        handle_max_lengths=(0.18, 0.09, 0.03),
        head_min_lengths=None,
        head_max_lengths=None,
    ),
]