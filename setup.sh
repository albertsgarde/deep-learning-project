LOG_PATH=$HOME/setup.log;

echo "" > $LOG_PATH;

echo "Installing curl and build-essential..."
apt-get update >> $LOG_PATH 2>&1;
apt-get install -y build-essential >> $LOG_PATH 2>&1;
apt-get install -y curl >> $LOG_PATH 2>&1;
apt-get update >> $LOG_PATH 2>&1;
echo "curl and build-essential installed."



echo "Installing Rust..." | tee -a $LOG_PATH;
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh -s -- -y >> $LOG_PATH 2>&1;
source "$HOME/.cargo/env" >> $LOG_PATH 2>&1;
echo "Rust installed." | tee -a $LOG_PATH;

cd $1/audio_samples_py;
echo "Installing maturin..." | tee -a $LOG_PATH;
pip install maturin >> $LOG_PATH 2>&1;
echo "Maturin installed." | tee -a $LOG_PATH;

echo "Building audio_samples_py..." | tee -a $LOG_PATH;
cargo clean >> $LOG_PATH 2>&1;
cargo update >> $LOG_PATH 2>&1;
maturin develop --release >> $LOG_PATH 2>&1;
echo "audio_samples_py built." | tee -a $LOG_PATH;

echo "Installing audio_samples_py..." | tee -a $LOG_PATH;
pip install --force-reinstall target/wheels/*.whl >> $LOG_PATH 2>&1;
echo "audio_samples_py installed." | tee -a $LOG_PATH;