from torch import nn
class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_size: int):
        super(Conv1DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),  # [B, 16, L/2]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, L/4]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # [B, 16, L/2]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),   # [B, 1, L]
            nn.Tanh(),  # since sine wave output is in [-1, 1]
        )

    def forward(self, x):
        # x: [B, 1, L]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Conv1DAutoencoder_test(nn.Module):
    def __init__(self, input_size: int):
        super(Conv1DAutoencoder_test, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),  # [B, 16, L/2]
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),  # [B, 32, L/4]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),  # [B, 16, L/2]
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=7, stride=2, padding=3, output_padding=1),   # [B, 1, L]
            nn.Tanh(),  # since sine wave output is in [-1, 1]
        )

    def forward(self, x):
        # x: [B, 1, L]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

