cd $1/audio_samples_py;
apt-get -y install cargo;
pip install maturin;
maturin build --sdist;
pip install target/wheels/*;