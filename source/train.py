import torch
import torchinfo
from utils import logs, config
from pathlib import Path
from model import Conv1DAutoencoder
import numpy as np
import datetime 
from omegaconf import OmegaConf
from hydra.utils import instantiate

def train_epoch(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0 
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("Batch_Loss/train", loss.item(), batch + epoch * len(dataloader))
        train_loss += loss.item()
        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /=  num_batches
    return train_loss

def test_epoch(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

def generate_audio_examples(model, device, dataloader):
    print("Running audio prediction...")
    prediction = torch.zeros(0).to(device)
    target = torch.zeros(0).to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            predicted_batch = model(X)
            prediction = torch.cat((prediction, predicted_batch.flatten()), 0)
            target = torch.cat((target, y.flatten()), 0)
    return prediction, target

def main():
    
    # Load the hyperparameters from the "params.yaml" file for usage with Tensorboard SummaryWriter
    params = config.Params()

    # Load the hyperparameters from the params yaml file 
    cfg = OmegaConf.load("params.yaml")

    # If defined in cfg use tensorboard path (define it in params.yaml for debugging purposes)
    # else use logs.return_tensorboard_path (default with dvc run)
    print(cfg.train.keys())
    if 'tensorboard_path' in cfg.train.keys():
        default_dir = cfg.train.tensorboard_path
        dvc_exp_name = 'debug'
        current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")

        tensorboard_path = Path(
            f"{default_dir}/logs/tensorboard/{current_datetime}_{dvc_exp_name}"
        )
    else:
        tensorboard_path = logs.return_tensorboard_path()

    # Create a SummaryWriter object to write the tensorboard logs
    metrics = {'Epoch_Loss/train': None, 'Epoch_Loss/test': None, 'Batch_Loss/train': None}
    if 'sync_interval' in cfg.train.keys():
        sync_interval = cfg.train.sync_interval
    else:
        sync_interval = None
    writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics, sync_interval=sync_interval)

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(cfg.general.random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.prepare_device(cfg.train.device_request)

    # Load preprocessed data from the input file into the training and testing tensors
    input_file_path = Path(cfg.train.input_file_path)
    data = torch.load(input_file_path, weights_only=False)
    X_training = data['X_training']
    y_training = data['y_training']
    X_testing = data['X_testing']
    y_testing = data['y_testing']

    # Create the model
    model = instantiate(cfg.model)
    summary = torchinfo.summary(model, (1, 1, cfg.model.input_size), device=device)
    print(summary)

    # Get example
    example_file_path = Path('data/raw/dummy_input_1.txt')
    example = np.loadtxt(example_file_path)
    example_tensor = torch.tensor(example, dtype=torch.float32)
    sample_inputs = example_tensor.unsqueeze(0)

    # # Add the model graph to the tensorboard logs
    #sample_inputs = torch.randn(1, 1, cfg.model.input_size) 
    writer.add_graph(model, sample_inputs.to(device))

    # Define the loss function and the optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    print(f'learning rate: {cfg.train.learning_rate}')

    # Create the dataloaders
    training_dataset = torch.utils.data.TensorDataset(X_training.unsqueeze(1), y_training.unsqueeze(1))
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    testing_dataset = torch.utils.data.TensorDataset(X_testing.unsqueeze(1), y_testing.unsqueeze(1))
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # Training loop
    for t in range(cfg.train.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_loss_train = train_epoch(training_dataloader, model, loss_fn, optimizer, device, writer, epoch=t)
        epoch_loss_test = test_epoch(testing_dataloader, model, loss_fn, device)
        epoch_audio_prediction, epoch_audio_target  = generate_audio_examples(model, device, testing_dataloader)
        print(epoch_loss_train)
        writer.add_scalar("Epoch_Loss/train", epoch_loss_train, t)
        writer.add_scalar("Epoch_Loss/test", epoch_loss_test, t)
        writer.add_audio("Audio/prediction", epoch_audio_prediction, t, sample_rate=44100)
        writer.add_audio("Audio/target", epoch_audio_target, t, sample_rate=44100)        
        writer.step()  

    writer.close()

    # Save the model checkpoint
    output_file_path = Path(cfg.train.output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.train.output_file_path)
    print("Saved PyTorch Model State to model.pth")

    print("Done with the training stage!")

if __name__ == "__main__":
    main()
