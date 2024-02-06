import numpy as np
import pretty_midi
import glob
import soundfile as sf
import matplotlib.pyplot as plt

# midi_fname = "/data5/kyungsu/drum_slayer/generated_data_prev/drum_data_train/generated_midi/kick_midi/kick_midi_1.midi"
# loop_fname = "/data5/kyungsu/drum_slayer/generated_data/drum_data_train/dafx_loops/1.wav"
gt_midi_fnames = glob.glob("results/*_gt.mid")
pred_midi_fnames = glob.glob("results/*_pred.mid")


def draw_drum_roll(notes):
    # Create a matrix of zeros, where each column corresponds to a unique pitch
    # and each row corresponds to a time frame
    drum_roll = np.zeros([3, 501])
    # Fill in the piano roll with the midi notes
    pitch_to_type = {36:0, 38:1, 42:2}
    for note in notes:
        pitch = int(note.pitch)
        type = pitch_to_type[pitch]
        start = int(note.start // 0.01)
        end = start + 2
        drum_roll[2-type, start:end] = 1
    return drum_roll

if __name__ == "__main__":
    for midi_fname in gt_midi_fnames+pred_midi_fnames:
        pm = pretty_midi.PrettyMIDI(midi_fname)
        # plot a piano roll
        drum_roll = draw_drum_roll(pm.instruments[0].notes)
        plt.figure(figsize=(12, 4))
        plt.imshow(drum_roll, aspect='auto', interpolation='nearest')
        # Put y ticks labels at the center of each cell
        plt.yticks(np.arange(3), ["Hihat", "Snare", "Kick"])
        # Put x ticks as 0 to 5 seconds in steps of one second
        plt.xticks(np.arange(0,501,100), np.arange(6))
        # Set axis labels
        plt.xlabel("Seconds")
        plt.savefig(f"{midi_fname.replace('.mid', '.png')}")
