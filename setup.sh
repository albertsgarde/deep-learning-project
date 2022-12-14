LOG_PATH=$HOME/setup.log;

echo "" > $LOG_PATH;

echo "Installing necessary linux packages..."
apt-get update >> $LOG_PATH 2>&1;
apt-get install -y build-essential >> $LOG_PATH 2>&1;
apt-get install -y curl >> $LOG_PATH 2>&1;
apt-get update >> $LOG_PATH 2>&1;
echo "Installed necessary linux packages."

echo "Installing necessary python packages..."
pip3 install matplotlib >> $LOG_PATH 2>&1;
pip3 install ipython >> $LOG_PATH 2>&1;
echo "Installed necessary python packages."

echo "Installing Rust..." | tee -a $LOG_PATH;
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh -s -- -y >> $LOG_PATH 2>&1;
source "$HOME/.cargo/env" >> $LOG_PATH 2>&1;
hash cargo | tee -a $LOG_PATH;
echo "Installed Rust." | tee -a $LOG_PATH;

cd $1/audio_samples_py;
echo "Installing maturin..." | tee -a $LOG_PATH;
pip3 install maturin >> $LOG_PATH 2>&1;
echo "Installed Maturin." | tee -a $LOG_PATH;

echo "Building audio_samples_py..." | tee -a $LOG_PATH;
hash cargo | tee -a $LOG_PATH;
cargo clean >> $LOG_PATH 2>&1;
cargo update >> $LOG_PATH 2>&1;
maturin build --release --sdist >> $LOG_PATH 2>&1;
echo "Built audio_samples_py." | tee -a $LOG_PATH;

echo "Installing audio_samples_py..." | tee -a $LOG_PATH;
pip3 uninstall -y audio_samples_py >> $LOG_PATH 2>&1;
pip3 install --force-reinstall target/wheels/*.whl >> $LOG_PATH 2>&1;
echo "Installed audio_samples_py." | tee -a $LOG_PATH;