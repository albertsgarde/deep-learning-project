cd $1/audio_samples_py;
pwd
ls
apt-get install cargo
export PATH=$PATH:$HOME/.cargo/bin
pip install maturin;
maturin build --sdist;
pwd
ls
pip install target/wheels/*;