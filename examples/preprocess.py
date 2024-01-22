import hydra
from omegaconf import DictConfig
from ssl4speechsynthesis.preprocessor.preprocess import Preprocessor
@hydra.main(version_base="1.3", config_name="config", config_path="config")
def main(cfg: DictConfig):
    preprocessor = Preprocessor(cfg.preprocess,split="valid")
    # preprocessor = Preprocessor(cfg.preprocess,split="train")
    preprocessor.preprocess()
    pass

if __name__ == "__main__":
    main()