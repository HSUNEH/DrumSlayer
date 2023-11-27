import dac
from audiotools import AudioSignal

import torch
# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

# device = torch.device("mps")
# model = model.to(device)


# Load audio signal file
signal = AudioSignal('CANTSAYSNARE.wav')
# signal.audio_data : torch.Size([1, 2, 9132])
# signal.sample_rate : 44100

# # Encode audio signal as one long file
# # (may run out of GPU memory on long files)
# signal.to(model.device)

# # mono_audio = torch.mean(signal.audio_data.squeeze(), dim=0, keepdim=True)
# x = model.preprocess(signal.audio_data, signal.sample_rate)

# x = torch.mean(model.preprocess(signal.audio_data, signal.sample_rate), dim=1, keepdim=True)

# # what is problem of signal
# z, codes, latents, _, _ = model.encode(x)



# # Decode audio signal
# y = model.decode(z)

### Alternatively, use the `compress` and `decompress` functions
# to compress long files.
signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)
# y = y / torch.max(torch.abs(y))


# Write to file
y.write('s_output.wav')

print("Done!")
