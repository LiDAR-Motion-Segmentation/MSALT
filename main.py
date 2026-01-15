import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.ui.main_window import MainWindow
from src.data.data_controller import DataController
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MSALT_Entry")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(f"Active Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setting this attribute changes that behavior to allow Resource Sharing with GPU for the 2 windows
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    
    app.setApplicationName(cfg.app.window_title)

    try:
        logger.info("Initializing Data Controller...")
        data_ctrl = DataController(cfg)

        total_frames = data_ctrl.get_total_frames()
        logger.info(f"Data Loaded Successfully. Total Frames: {total_frames}")

        if total_frames == 0:
            logger.warning(
                "Dataset appears empty. Check your paths in conf/msalt_setup/..."
            )

    except Exception as e:
        logger.critical(f"Failed to initialize Data Controller: {e}", exc_info=True)
        sys.exit(1)

    try:
        window = MainWindow(data_controller=data_ctrl)
        window.show()

        # Maximize for "Production Tool" feel
        window.showMaximized()

    except Exception as e:
        logger.critical(f"Failed to launch UI: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    if not Path("config").exists():
        print("CRITICAL ERROR: 'conf' directory not found.")
        print("Please run this script from the root of the repository.")
        sys.exit(1)
    main()
