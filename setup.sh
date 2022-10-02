mkdir downloads;
cd downloads;
wget https://static.rust-lang.org/dist/rust-1.64.0-x86_64-unknown-linux-gnu.tar.gz;
tar -xzf rust-1.64.0-x86_64-unknown-linux-gnu.tar.gz;
sudo ./rust-1.64.0-x86_64-unknown-linux-gnu/install.sh;
cd ..;
cd $1/audio_samples_py;
pwd
ls

#apt-get -y install rustc &> ../../setup.log ;
#echo "rustc installed"
#echo "rustc installed" > ../../setup.log ;
#apt-get -y install cargo &> ../../setup.log ;
#echo "cargo installed" 
#echo "cargo installed" >> ../../setup.log ;
#export PATH=$PATH:$HOME/.cargo/bin
pip install maturin >> ../../setup.log 2>&1 ;
echo "maturin installed"
echo "maturin installed" >> ../../setup.log ;
maturin build --sdist >> ../../setup.log 2>&1 ;
echo "package built"
echo "package built" >> ../../setup.log ;
pwd
ls
pip install target/wheels/* >> ../../setup.log 2>&1 ;