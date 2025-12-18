from pathlib import Path
from omegaconf import OmegaConf


def check_setup():
    print("--- 1. FILE SYSTEM CHECK ---")
    root = Path("config")

    # Check Main Config
    if not (root / "config.yaml").exists():
        print("CRITICAL: config/config.yaml is MISSING!")
    else:
        print("OK: config/config.yaml found.")

    # Check Models Folder
    models_dir = root / "models"
    if not models_dir.exists():
        print(
            f"CRITICAL: Directory 'config/models' does not exist. Found: {list(root.glob('*'))}"
        )
        return

    # Check Default Model File
    model_file = models_dir / "default.yaml"
    if not model_file.exists():
        print(
            f"CRITICAL: 'config/models/default.yaml' MISSING! Files in models/: {list(models_dir.glob('*'))}"
        )
        return

    print("OK: config/models/default.yaml found.")

    print("\n--- 2. CONTENT CHECK ---")
    try:
        conf = OmegaConf.load(model_file)
        print(f"Raw content of default.yaml:\n{conf}")
        if "segmentation" not in conf:
            print(
                "CRITICAL ERROR: 'segmentation' key missing from default.yaml. Check indentation!"
            )
        else:
            print("OK: Key 'segmentation' found.")
    except Exception as e:
        print(f"CRITICAL: Could not parse default.yaml: {e}")


if __name__ == "__main__":
    check_setup()
