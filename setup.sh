cd $1/audio_samples_py;
source .env/bin/activate;
pip install maturin;
maturin build --sdist;