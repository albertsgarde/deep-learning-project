LOG_PATH=$HOME/setup.log;

echo "" > $LOG_PATH;

apt-get update>> $LOG_PATH 2>&1;
if hash build-essential &> /dev/null; then
    echo "build-essential is installed." | tee -a $LOG_PATH;
else
    echo "Installing build-essential..." | tee -a $LOG_PATH;
    apt-get install -y build-essential >> $LOG_PATH 2>&1;
    apt-get update >> $LOG_PATH 2>&1;
    echo "build-essential installed." | tee -a $LOG_PATH;
fi
if hash curl &> /dev/null; then
    echo "curl is installed." | tee -a $LOG_PATH;
else
    echo "Installing curl..." | tee -a $LOG_PATH;
    apt-get install -y curl >> $LOG_PATH 2>&1;
    apt-get update >> $LOG_PATH 2>&1;
    echo "curl installed." | tee -a $LOG_PATH;
fi
if hash cargo &> /dev/null
then 
    echo "Rust already installed." | tee -a $LOG_PATH;
else 
    echo "Installing Rust..." | tee -a $LOG_PATH;
    curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh -s -- -y >> $LOG_PATH 2>&1;
    echo "Rust installed." | tee -a $LOG_PATH;
fi

cd $1/audio_samples_py;
if pip list | grep -F maturin &> /dev/null
then 
    echo "Maturin already installed" | tee -a $LOG_PATH;
else 
    echo "Installing maturin..." | tee -a $LOG_PATH;
    pip install maturin >> $LOG_PATH 2>&1;
    echo "Maturin installed." | tee -a $LOG_PATH;
    
fi

ls
cargo clean
ls
echo "Building audio_samples_py..." | tee -a $LOG_PATH;
maturin build --release >> $LOG_PATH 2>&1;
echo "audio_samples_py built." | tee -a $LOG_PATH;

echo "Installing audio_samples_py..." | tee -a $LOG_PATH;
pip install --force-reinstall target/wheels/*.whl >> $LOG_PATH 2>&1;
echo "audio_samples_py installed." | tee -a $LOG_PATH;