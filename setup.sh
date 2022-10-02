cd $1/audio_samples_py;
source .env/bin/activate;
apt-get install cargo;
pip install maturin;
maturin build --sdist;
deactivate;
pip install target/wheels/*;