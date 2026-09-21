from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DATA = ROOT / "data"

FILES = {
    "uhpfrc_ml": DATA / "ml" / "uhpfrc_compressive_strength_database.xlsx",

    "moment_curvature": (
        DATA / "experimental" / "beam_moment_curvature_results.xlsx"
    ),
    "load_displacement_ductility": (
        DATA / "experimental" / "beam_load_displacement_ductility.xlsx"
    ),
    "beam_experiments": (
        DATA / "experimental" / "beam_experimental_load_displacement.xlsx"
    ),
    "control_beam_strain": (
        DATA / "experimental" / "control_beam_strain_results.xlsx"
    ),
    "strain_response": (
        DATA / "experimental" / "beam_strain_response_results.xlsx"
    ),
    "retrofitted_load_displacement": (
        DATA / "experimental" / "retrofitted_beam_load_displacement.xlsx"
    ),

    "experimental_numerical_validation": (
        DATA / "finite_element" / "experimental_numerical_validation.xlsx"
    ),
    "cdp_properties": (
        DATA / "finite_element" / "concrete_damage_plasticity_properties.xlsx"
    ),
    "mesh_sensitivity": (
        DATA / "finite_element" / "finite_element_mesh_sensitivity.xlsx"
    ),

    "raw_channels_1": (
        DATA / "raw_test_data" / "experimental_displacement_channels.csv"
    ),
    "raw_channels_2": (
        DATA / "raw_test_data" / "experimental_load_strain_channels.csv"
    ),
}


def run(command, title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print("Command:", " ".join(str(x) for x in command))

    completed = subprocess.run(
        [str(x) for x in command],
        cwd=ROOT,
    )

    if completed.returncode == 0:
        print(f"[OK] {title}")
    else:
        print(f"[FAILED] {title}")

    return completed.returncode


def available(*keys):
    missing = [FILES[key] for key in keys if not FILES[key].exists()]

    if not missing:
        return True

    print("\n[SKIPPED] Missing required input:")
    for path in missing:
        print("  -", path.relative_to(ROOT))
    return False


def run_uhpfrc_ml():
    if not available("uhpfrc_ml"):
        return

    src = ROOT / "projects" / "uhpfrc-compressive-strength" / "src"

    run(
        [PYTHON, src / "exploratory_analysis.py"],
        "UHPFRC exploratory analysis",
    )

    run(
        [PYTHON, src / "feature_selection.py"],
        "UHPFRC feature selection",
    )

    model_scripts = [
        "linear_regression.py",
        "svr.py",
        "random_forest.py",
        "gradient_boosting.py",
        "decision_tree.py",
        "xgboost_model.py",
        "lightgbm_model.py",
        "mlp.py",
    ]

    for script in model_scripts:
        run(
            [PYTHON, src / script],
            f"Train model: {Path(script).stem}",
        )

    run(
        [PYTHON, src / "compare_models.py"],
        "Compare ML models",
    )

    run(
        [PYTHON, src / "actual_predicted_series.py"],
        "Actual vs predicted series",
    )

    run(
        [PYTHON, src / "taylor_diagram.py"],
        "Taylor diagram",
    )

    for model in [
        "random_forest",
        "gradient_boosting",
        "decision_tree",
        "xgboost",
        "lightgbm",
        "mlp",
    ]:
        run(
            [PYTHON, src / "shap_analysis.py", "--model", model],
            f"SHAP analysis: {model}",
        )

        run(
            [PYTHON, src / "lime_analysis.py", "--model", model],
            f"LIME analysis: {model}",
        )

        run(
            [PYTHON, src / "feature_importance.py", "--model", model],
            f"Feature importance: {model}",
        )


def run_supporting_analysis():
    src = ROOT / "supporting-analysis"

    if available("moment_curvature"):
        run(
            [
                PYTHON,
                src / "moment_curvature.py",
                FILES["moment_curvature"],
                "--sheet",
                "Sheet3",
            ],
            "Moment-curvature analysis",
        )

    if available("load_displacement_ductility"):
        run(
            [
                PYTHON,
                src / "load_displacement_analysis.py",
                FILES["load_displacement_ductility"],
            ],
            "Load-displacement / ductility analysis",
        )

    if available("beam_experiments"):
        run(
            [
                PYTHON,
                src / "beam_experiments.py",
                FILES["beam_experiments"],
            ],
            "Beam experimental load-displacement analysis",
        )

    if available("strain_response"):
        run(
            [
                PYTHON,
                src / "strain_response.py",
                FILES["strain_response"],
                "--sheet",
                "Sheet2",
            ],
            "Beam strain-response analysis",
        )

    if available("experimental_numerical_validation"):
        run(
            [
                PYTHON,
                src / "experimental_numerical_validation.py",
                FILES["experimental_numerical_validation"],
            ],
            "Experimental vs numerical validation",
        )

    if available("cdp_properties"):
        run(
            [
                PYTHON,
                src / "cdp_properties.py",
                FILES["cdp_properties"],
                "--sheet",
                "Sheet3",
            ],
            "Concrete damage plasticity plots",
        )

    if available("mesh_sensitivity"):
        run(
            [
                PYTHON,
                src / "mesh_sensitivity.py",
                FILES["mesh_sensitivity"],
            ],
            "FE mesh-sensitivity analysis",
        )

    if available("raw_channels_1", "raw_channels_2"):
        run(
            [
                PYTHON,
                src / "raw_channel_processing.py",
                FILES["raw_channels_1"],
                FILES["raw_channels_2"],
                "--output",
                DATA / "raw_test_data" / "processed_channels.csv",
            ],
            "Raw experimental channel processing",
        )

    # These use hard-coded research values and need no external file.
    run(
        [PYTHON, src / "jacket_thickness_capacity.py"],
        "Jacket-thickness capacity plot",
    )

    run(
        [PYTHON, src / "material_test_plots.py"],
        "Material-test plots",
    )


def print_file_status():
    print("\nExpected datasets")
    print("-" * 78)

    for key, path in FILES.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(
            f"{status:8} | "
            f"{key:34} | "
            f"{path.relative_to(ROOT)}"
        )


def main():
    print("Structural Engineering Analytics — Master Runner")
    print("Repository:", ROOT)

    print_file_status()

    run_uhpfrc_ml()
    run_supporting_analysis()

    print("\n" + "=" * 78)
    print("Finished.")
    print("Any analysis with a missing input file was skipped.")
    print("=" * 78)


if __name__ == "__main__":
    main()
