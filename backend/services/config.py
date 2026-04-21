import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_PATH = BASE_DIR / "config"
    ICON_OUTPUT_PATH = BASE_DIR / "temp"

    def __init__(self) -> None:
        self.load_from_folder()

    def load_from_folder(self):
        """
        Load all YAML configuration files from the CONFIG_PATH directory and add them as attributes to the Config instance.
        """
        if not self.CONFIG_PATH.exists():
            # Fallback for different working directory layouts
            fallback_path = Path().resolve() / "config"
            if fallback_path.exists():
                self.CONFIG_PATH = fallback_path
            
        logger.info(f"Loading configuration from {self.CONFIG_PATH}")
        config_files = self.CONFIG_PATH.glob("*.yaml")
        loaded_configs = []
        for config_file in config_files:
            config_name = config_file.stem
            try:
                with open(config_file, 'r') as file:
                    config_data = yaml.safe_load(file)
                setattr(self, config_name, config_data)
                loaded_configs.append(config_name)
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
        
        logger.info(f"Loaded configurations: {', '.join(loaded_configs)}")


if __name__ == "__main__":
    config = Config()
    breakpoint()  # For debugging purposes