cd $1/audio_samples_py;
pwd
ls
apt-get -y install cargo &> ../setup.log ;
export PATH=$PATH:$HOME/.cargo/bin
pip install maturin >> ../setup.log 2>&1 ;
echo "maturin installed"
maturin build --sdist >> ../setup.log 2>&1 ;
echo "package built"
pwd
ls
pip install target/wheels/* >> ../setup.log 2>&1 ;