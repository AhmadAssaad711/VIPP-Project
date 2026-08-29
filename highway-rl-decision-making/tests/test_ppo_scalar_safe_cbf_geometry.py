from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ppo_reward_safety import install_scalar_safe_cbf_geometry


def test_scalar_safe_ellipse_radius_matches_notebook_formula() -> None:
    namespace: dict[str, object] = {"CBF_EPS_SIDE": 0.10}
    install_scalar_safe_cbf_geometry(namespace)
    safe_radius = namespace["ellipse_radius_along_line"]

    rng = np.random.default_rng(307)
    for a, b, delta in zip(
        rng.uniform(1.0, 6.0, 20_000),
        rng.uniform(0.5, 3.0, 20_000),
        rng.uniform(-math.pi, math.pi, 20_000),
    ):
        expected = float(
            a
            * b
            / max(
                float(
                    np.sqrt(
                        (b * np.cos(float(delta))) ** 2
                        + (a * np.sin(float(delta))) ** 2
                    )
                ),
                1e-9,
            )
        )
        observed = safe_radius(float(a), float(b), float(delta))
        assert math.isclose(observed, expected, rel_tol=1e-14, abs_tol=1e-14)


def test_notebook_downstream_function_resolves_safe_geometry() -> None:
    namespace: dict[str, object] = {"CBF_EPS_SIDE": 0.10}
    exec(
        "def downstream(p, ego, other):\n"
        "    return pairwise_centerline_clearance(p, ego, other, CBF_EPS_SIDE)\n",
        namespace,
    )
    install_scalar_safe_cbf_geometry(namespace)

    ego = {"length": 3.5, "width": 1.8, "heading": 0.15}
    other = {"length": 4.2, "width": 2.0, "heading": -0.25}
    point = np.asarray([12.5, -2.3], dtype=float)
    observed = namespace["downstream"](point, ego, other)

    radius = float(np.linalg.norm(point))
    phi = float(np.arctan2(point[1], point[0]))
    axes = []
    for vehicle in (ego, other):
        axes.append(
            (
                float(vehicle["length"]) / np.sqrt(2.0) + 0.2,
                float(vehicle["width"]) / np.sqrt(2.0) + 0.2,
            )
        )
    projected = []
    for (a, b), vehicle in zip(axes, (ego, other)):
        delta = (phi - float(vehicle["heading"]) + np.pi) % (2.0 * np.pi) - np.pi
        projected.append(
            float(a * b / np.sqrt((b * np.cos(delta)) ** 2 + (a * np.sin(delta)) ** 2))
        )
    expected = (radius - sum(projected), radius, projected[0], projected[1])

    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-14)
    assert namespace["CBF_SCALAR_GEOMETRY_BACKEND"] == "python_math_v1"


def test_kpi_geometry_aliases_use_the_same_safe_scalar_backend() -> None:
    namespace: dict[str, object] = {
        "CBF_EPS_SIDE": 0.12,
        "_kpi_base_env": lambda env: env,
    }
    install_scalar_safe_cbf_geometry(namespace)

    assert namespace["_kpi_inflated_axes"] is namespace["inflated_ellipse_axes"]
    assert namespace["_kpi_wrap_angle"] is namespace["_wrap_angle"]
    assert namespace["_kpi_ellipse_radius"] is namespace["ellipse_radius_along_line"]

    a, b = namespace["_kpi_inflated_axes"](3.5, 1.8, 0.12)
    observed = namespace["_kpi_ellipse_radius"](a, b, 0.37)
    expected = float(
        a
        * b
        / np.sqrt(
            (b * np.cos(0.37)) ** 2
            + (a * np.sin(0.37)) ** 2
        )
    )
    assert math.isclose(observed, expected, rel_tol=1e-14, abs_tol=1e-14)
