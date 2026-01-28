import yaml
from pathlib import Path

class Config:
    CONFIG_PATH = Path().resolve() / "config"
    ICON_OUTPUT_PATH = Path().resolve() / "temp"

    def __init__(self) -> None:
        self.load_from_folder()

    def load_from_folder(self):
        """
        Load all YAML configuration files from the CONFIG_PATH directory and add them as attributes to the Config instance.
        """
        config_files = Path(self.CONFIG_PATH).glob("*.yaml")
        for config_file in config_files:
            config_name = config_file.stem
            with open(config_file, 'r') as file:
                config_data = yaml.safe_load(file)
            setattr(self, config_name, config_data)


if __name__ == "__main__":
    config = Config()
    breakpoint()  # For debugging purposes