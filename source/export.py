import torch
from utils import config
from model import Conv1DAutoencoder #NeuralNetwork
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

def main():
    # Load the hyperparameters from the params yaml file 
    cfg = OmegaConf.load("params.yaml")
    print(cfg)
    # Define the model (ensure to use the same architecture as in train.py)
    #model = Conv1DAutoencoder(cfg.model.input_size)
    model = instantiate(cfg.model)

    # Load the model state
    input_file_path = Path('models/checkpoints/model.pth')
    model.load_state_dict(torch.load(input_file_path, map_location=torch.device('cpu')))

    # Export the model
    #example = torch.rand(1, 1, input_size)
    output_file_path = Path(cfg.export.output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_file_path)
    print("Model exported to .pth format.")

if __name__ == "__main__":
    main()
