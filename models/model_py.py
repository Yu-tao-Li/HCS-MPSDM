"""OpenSeesPy implementation of the calibrated three-story SAC frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import openseespy.opensees as ops


@dataclass(frozen=True)
class ModelConfig:
    mass_scale: float = 1.0
    stiffness_scale: float = 1.071
    strength_scale: float = 0.76
    ductility_scale: float = 0.42
    story_2_stiffness_modifier: float = 1.0
    story_3_stiffness_modifier: float = 1.0


SECTIONS_IMPERIAL = {
    # A (in2), Ix (in4), Zx (in3), depth (in), h/tw, bf/(2tf), ry (in)
    "W14X257": (75.6, 3400.0, 487.0, 16.4, 9.71, 4.23, 4.13),
    "W14X311": (91.4, 4330.0, 603.0, 17.1, 8.09, 3.59, 4.20),
    "W33X118": (34.7, 5900.0, 415.0, 32.9, 54.5, 7.76, 2.32),
    "W30X116": (34.2, 4930.0, 378.0, 30.0, 47.8, 6.17, 2.19),
    "W24X68": (20.1, 1830.0, 177.0, 23.7, 52.0, 7.66, 1.87),
}


def section_database() -> dict[str, dict[str, float]]:
    converted: dict[str, dict[str, float]] = {}
    for name, row in SECTIONS_IMPERIAL.items():
        area, inertia, plastic_modulus, depth, htw, bftf, ry = row
        converted[name] = {
            "A": area * 645.16,
            "I": inertia * 25.4**4,
            "Z": plastic_modulus * 25.4**3,
            "d": depth * 25.4,
            "htw": htw,
            "bftf": bftf,
            "ry": ry * 25.4,
        }
    return converted


def make_imk_hinge(
    tag: int,
    node_i: int,
    node_j: int,
    elastic_modulus: float,
    inertia: float,
    length: float,
    yield_moment: float,
    theta_p: float,
    theta_pc: float,
    deterioration_lambda: float,
) -> None:
    n = 100.0
    stiffness = (n + 1.0) * 6.0 * elastic_modulus * inertia / length
    theta_u = 0.20
    capping_ratio = 1.13
    residual_ratio = 0.30
    ops.uniaxialMaterial(
        "IMKBilin",
        tag,
        stiffness,
        theta_p,
        theta_pc,
        theta_u,
        yield_moment,
        capping_ratio,
        residual_ratio,
        theta_p,
        theta_pc,
        theta_u,
        yield_moment,
        capping_ratio,
        residual_ratio,
        deterioration_lambda,
        deterioration_lambda,
        deterioration_lambda,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    ops.element(
        "zeroLength",
        tag,
        node_i,
        node_j,
        "-mat",
        99,
        99,
        tag,
        "-dir",
        1,
        2,
        6,
        "-doRayleigh",
        1,
    )


def build_model(config: ModelConfig = ModelConfig()) -> dict[str, object]:
    """Build the model and complete gravity analysis."""

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    n_story = 3
    n_bay = 3
    story_height = 3960.0
    bay_width = 9150.0
    gravity = 9810.0
    elastic_modulus = 200000.0 * config.stiffness_scale
    fy_beam = 339.0
    fy_column = 397.0
    n_spring = 100.0
    story_k = {
        1: 1.0,
        2: config.story_2_stiffness_modifier,
        3: config.story_3_stiffness_modifier,
    }
    sections = section_database()
    floor_mass = {
        1: 487.5 * config.mass_scale,
        2: 487.5 * config.mass_scale,
        3: 530.0 * config.mass_scale,
    }

    for story in range(n_story + 1):
        for axis in range(n_bay + 1):
            tag = story * 10 + axis + 1
            ops.node(tag, axis * bay_width, story * story_height)
            if story == 0:
                ops.fix(tag, 1, 1, 1)

    leaning_x = 4.0 * bay_width
    for story in range(n_story + 1):
        tag = 9000 + story
        ops.node(tag, leaning_x, story * story_height)
        if story == 0:
            ops.fix(tag, 1, 1, 1)
        else:
            ops.fix(tag, 0, 0, 1)

    for story in range(1, n_story + 1):
        master = story * 10 + 1
        for axis in range(1, n_bay + 1):
            ops.equalDOF(master, story * 10 + axis + 1, 1)
        ops.equalDOF(master, 9000 + story, 1)
        ops.mass(master, floor_mass[story], 1.0e-9, 1.0e-9)

    ops.geomTransf("PDelta", 1)
    ops.geomTransf("Linear", 2)
    ops.uniaxialMaterial("Elastic", 99, 1.0e15)

    element_tag = 1000
    spring_tag = 100000
    for story in range(1, n_story + 1):
        for axis in range(n_bay + 1):
            if axis in (0, n_bay):
                section_name = "W14X257"
                axial_ratio = 0.15
            else:
                section_name = "W14X311"
                axial_ratio = 0.25
            section = sections[section_name]
            joint_bottom = (story - 1) * 10 + axis + 1
            joint_top = story * 10 + axis + 1
            node_bottom = 10000 + story * 100 + axis * 10 + 1
            node_top = 10000 + story * 100 + axis * 10 + 2
            ops.node(node_bottom, axis * bay_width, (story - 1) * story_height)
            ops.node(node_top, axis * bay_width, story * story_height)

            if axial_ratio <= 0.20:
                axial_reduction = 1.15 * (1.0 - axial_ratio / 2.0)
            else:
                axial_reduction = 1.15 * (9.0 / 8.0) * (1.0 - axial_ratio)
            yield_moment = (
                fy_column
                * section["Z"]
                * axial_reduction
                * config.strength_scale
            )
            theta_p = 0.040 * config.ductility_scale
            theta_pc = 0.350
            calibrated_inertia = section["I"] * story_k[story]
            spring_tag += 1
            make_imk_hinge(
                spring_tag,
                joint_bottom,
                node_bottom,
                elastic_modulus,
                calibrated_inertia,
                story_height,
                yield_moment,
                theta_p,
                theta_pc,
                2.0,
            )
            spring_tag += 1
            make_imk_hinge(
                spring_tag,
                node_top,
                joint_top,
                elastic_modulus,
                calibrated_inertia,
                story_height,
                yield_moment,
                theta_p,
                theta_pc,
                2.0,
            )
            element_tag += 1
            modified_inertia = calibrated_inertia * (n_spring + 1.0) / n_spring
            ops.element(
                "elasticBeamColumn",
                element_tag,
                node_bottom,
                node_top,
                section["A"],
                elastic_modulus,
                modified_inertia,
                1,
            )

    beam_sections = {1: "W33X118", 2: "W30X116", 3: "W24X68"}
    for story in range(1, n_story + 1):
        section = sections[beam_sections[story]]
        for bay in range(n_bay):
            joint_left = story * 10 + bay + 1
            joint_right = story * 10 + bay + 2
            node_left = 20000 + story * 100 + bay * 10 + 1
            node_right = 20000 + story * 100 + bay * 10 + 2
            ops.node(node_left, bay * bay_width, story * story_height)
            ops.node(node_right, (bay + 1) * bay_width, story * story_height)

            yield_moment = (
                1.17 * fy_beam * section["Z"] * config.strength_scale
            )
            theta_p = 0.030 * config.ductility_scale
            theta_pc = 0.280
            calibrated_inertia = section["I"] * story_k[story]
            spring_tag += 1
            make_imk_hinge(
                spring_tag,
                joint_left,
                node_left,
                elastic_modulus,
                calibrated_inertia,
                bay_width,
                yield_moment,
                theta_p,
                theta_pc,
                2.0,
            )
            spring_tag += 1
            make_imk_hinge(
                spring_tag,
                node_right,
                joint_right,
                elastic_modulus,
                calibrated_inertia,
                bay_width,
                yield_moment,
                theta_p,
                theta_pc,
                2.0,
            )
            element_tag += 1
            modified_inertia = calibrated_inertia * (n_spring + 1.0) / n_spring
            ops.element(
                "elasticBeamColumn",
                element_tag,
                node_left,
                node_right,
                section["A"],
                elastic_modulus,
                modified_inertia,
                2,
            )

    ops.uniaxialMaterial("Elastic", 500, 200000.0)
    for story in range(1, n_story + 1):
        ops.element(
            "corotTruss",
            8000 + story,
            9000 + story - 1,
            9000 + story,
            1.0e6,
            500,
        )

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for story in range(1, n_story + 1):
        ops.load(9000 + story, 0.0, -floor_mass[story] * gravity, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-8, 50, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    status = ops.analyze(10)
    if status != 0:
        raise RuntimeError(f"Gravity analysis failed with code {status}")
    ops.loadConst("-time", 0.0)

    return {
        "story_height": story_height,
        "bay_width": bay_width,
        "gravity": gravity,
        "floor_mass": floor_mass,
        "master_nodes": [11, 21, 31],
        "base_nodes": [1, 2, 3, 4, 9000],
    }


def modal_analysis(config: ModelConfig = ModelConfig()) -> dict[str, list[float]]:
    metadata = build_model(config)
    eigenvalues = ops.eigen("-fullGenLapack", 3)
    periods = [2.0 * math.pi / math.sqrt(value) for value in eigenvalues]

    masses = metadata["floor_mass"]
    total_mass = sum(masses.values())
    ratios: list[float] = []
    for mode in range(1, 4):
        mode_shape = [
            ops.nodeEigenvector(node, mode, 1)
            for node in metadata["master_nodes"]
        ]
        numerator = sum(masses[i + 1] * mode_shape[i] for i in range(3))
        denominator = sum(
            masses[i + 1] * mode_shape[i] ** 2 for i in range(3)
        )
        effective_mass = numerator**2 / denominator
        ratios.append(100.0 * effective_mass / total_mass)
    return {
        "eigenvalue": list(eigenvalues),
        "period_s": periods,
        "modal_mass_ratio_percent": ratios,
    }


def pushover_analysis(
    config: ModelConfig = ModelConfig(),
    displacement_increment: float = 0.5,
) -> dict[str, list[float] | int]:
    metadata = build_model(config)
    story_height = float(metadata["story_height"])
    floor_mass = metadata["floor_mass"]
    total_weight = sum(floor_mass.values()) * float(metadata["gravity"])

    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(11, floor_mass[1] * 1.0, 0.0, 0.0)
    ops.load(21, floor_mass[2] * 2.0, 0.0, 0.0)
    ops.load(31, floor_mass[3] * 3.0, 0.0, 0.0)

    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-6, 100, 0)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", 31, 1, displacement_increment)
    ops.analysis("Static")

    target_displacement = 0.05 * 3.0 * story_height
    drift: list[float] = []
    normalized_shear: list[float] = []
    status = 0
    while ops.nodeDisp(31, 1) < target_displacement and status == 0:
        status = ops.analyze(1)
        if status != 0:
            ops.test("NormDispIncr", 1.0e-5, 300, 0)
            ops.algorithm("NewtonLineSearch", "-type", "Bisection")
            status = ops.analyze(1)
        if status != 0:
            ops.algorithm("KrylovNewton")
            status = ops.analyze(1)
        if status == 0:
            ops.algorithm("Newton")
            ops.test("NormDispIncr", 1.0e-6, 100, 0)
            ops.reactions()
            base_shear = -sum(
                ops.nodeReaction(node, 1) for node in metadata["base_nodes"]
            )
            roof_displacement = ops.nodeDisp(31, 1)
            drift.append(roof_displacement / (3.0 * story_height))
            normalized_shear.append(base_shear / total_weight)
    return {
        "status": status,
        "roof_drift": drift,
        "normalized_base_shear": normalized_shear,
    }
