# DrumSlayer
The End of Drum Reconstruction

Mixture Audio -> Drum Midi & Drum Source

## Dataset
requirements 
1. `pip install -r requirements.txt`
2. `pip install -r data_generate/DAFXChain/requirements.txt`
3. extra install in `data_generate/DAFXChain/requirements.txt`

### Make your dataset by yourself

1. `scp -rP 20022 marg@147.47.120.221:/home/marg/st_drums/one_shots <one shot dir>` 

2. `python data_generate/data_generate.py --data_type all --oneshot_dir <one shot dir> --output_dir <dir to save>`
