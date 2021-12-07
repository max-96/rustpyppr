from typing import Dict, List


def forward_push(edge_dict: Dict[int, List[int]],
                     source: int,
                     damping_factor: float,
                     r_max: float) -> Dict[int, float]:
    ...


def forward_push_vec(edge_dict: Dict[int, List[int]],
                         source: int,
                         damping_factor: float,
                         r_max: float) -> Dict[int, float]:
    ...


def forward_push_vec_lazy(edge_dict: Dict[int, List[int]],
                              source: int,
                              damping_factor: float,
                              r_max: float) -> Dict[int, float]:
    ...


def multiple_forward_push_vec(edge_dict: Dict[int, List[int]],
                                  sources: List[int],
                                  damping_factor: float,
                                  r_max: float) -> Dict[int, Dict[int, float]]:
    ...


def multiple_forward_push_vec_lazy(edge_dict: Dict[int, List[int]],
                                       sources: List[int],
                                       damping_factor: float,
                                       r_max: float) -> Dict[int, Dict[int, float]]:
    ...
