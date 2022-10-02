cd $1/audio_samples_py;
pwd
ls
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin;
maturin build --sdist;
pwd
ls
pip install target/wheels/*;