python -m gdown "https://drive.google.com/uc?id=1FqN-pa955Wvu3utGViUKiVfza6cL_W0D" # CLaTr ckpt
python -m gdown "https://drive.google.com/uc?id=1uYeK1WcS3XI4uewHqi79RmLPggdpWgnG" # DIRECTOR ckpts

unzip director.zip
rm director.zip

mkdir checkpoints
mv clatr-e100.ckpt checkpoints
mv director checkpoints
