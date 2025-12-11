import sys
import hydra
from omegaconf import DictConfig
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import AnnotationToolWindow
from src.data.data_controller import DataController

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    app = QApplication(sys.argv)
    
    try:
        data_ctrl = DataController(cfg)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    cam_ids = [cam.id for cam in cfg.sensor_setup.cameras]
    window = AnnotationToolWindow(camera_names=cam_ids, data_controller=data_ctrl)
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()