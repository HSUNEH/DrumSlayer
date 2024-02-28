from encoder_decoder import EncoderDecoderModule, EncoderDecoderConfig    
from dataset import DrumSlayerDataset
from torch.utils.data import DataLoader
import wandb
import lightning.pytorch as pl
import os
import audiofile as af
import pretty_midi
import matplotlib.pyplot as plt
from investigate import draw_drum_roll

def main(ckpt_dir,data_dir):
    config = EncoderDecoderConfig()
    module = EncoderDecoderModule.load_from_checkpoint(ckpt_dir + "/2023-12-20-11_real/epoch=0-val_loss=1.62.ckpt", config=config)
    model = module.encoder_decoder
    model = model.cuda()
    test_dataset = DrumSlayerDataset(data_dir, "test")
    evaluate(model, test_dataset)

def evaluate(model, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    for i, batch in enumerate(test_dataloader):
        x, y = batch
        x = x.cuda()
        # y_hat = model(x, strategy="greedy")
        y_hat = model(x, strategy="top-p", sample_arg=0.5)

        y = y.detach().numpy()
        midi_gt = convert_to_midi(y)
        midi_gt.write(f"results/{i:03d}_gt.mid")
        plot_midi(midi_gt, f"results/{i:03d}_gt.png")

        y_hat = y_hat.cpu().detach().numpy()
        midi_pred = convert_to_midi(y_hat)
        midi_pred.write(f"results/{i:03d}_pred.mid")
        plot_midi(midi_pred, f"results/{i:03d}_pred.png")

        audio = af.read(data_dir + f"drum_data_test/dafx_loops/{i}.wav")[0]
        af.write(f"results/{i:03d}.wav", audio, 44100)
        import pdb; pdb.set_trace()
    
def convert_to_midi(tokens):
    tokens = tokens[0,1:-1]
    num_notes = tokens.shape[0] // 3
    drum_midi = pretty_midi.PrettyMIDI()
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    type_to_pitch = [36, 38, 42]
    for i in range(num_notes):
        onset = tokens[i*3] - 3
        vel = tokens[i*3+1] - 3 - 1000
        type = tokens[i*3+2] - 3 - 1000 - 128
        if type not in [0, 1, 2]:
            break
        drum_inst.notes.append(pretty_midi.Note(
            velocity=vel,
            pitch=type_to_pitch[type],
            start=onset*0.005,
            end=(onset+1)*0.005
        ))
    drum_midi.instruments.append(drum_inst)
    return drum_midi

def plot_midi(midi, fname):
    drum_roll = draw_drum_roll(midi.instruments[0].notes)
    plt.figure(figsize=(12, 4))
    plt.imshow(drum_roll, aspect='auto', interpolation='nearest')
    plt.savefig(fname)




if __name__ == "__main__":
    ckpt_dir = '/workspace/DrumTranscriber/ckpts/'
    data_dir = 'disk2/st_drums/generated_data/'
    main(ckpt_dir,data_dir)

    
    