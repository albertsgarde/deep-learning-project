LOG_PATH=../../setup.log;

echo "" > $LOG_PATH;

if hash cargo &> /dev/null
then 
    echo "Rust already installed." | tee $LOG_PATH;
else 
    mkdir -p downloads;
    cd downloads;
    echo "Installing Rust..." | tee $LOG_PATH;
    wget https://static.rust-lang.org/dist/rust-1.64.0-x86_64-unknown-linux-gnu.tar.gz >> $LOG_PATH 2>&1;;
    tar -xzf rust-1.64.0-x86_64-unknown-linux-gnu.tar.gz >> $LOG_PATH 2>&1;;
    ./rust-1.64.0-x86_64-unknown-linux-gnu/install.sh >> $LOG_PATH 2>&1;;
    cd ..;
    rm -rf downloads;
    echo "Rust installed." | tee $LOG_PATH;
fi

cd $1/audio_samples_py;
if pip list | grep -F maturin &> /dev/null
then 
    echo "Maturin already installed" | tee $LOG_PATH;
else 
    echo "Installing maturin..." | tee $LOG_PATH;
    pip install maturin >> $LOG_PATH 2>&1;
    echo "Maturin installed." | tee $LOG_PATH;
    
fi

echo "Building audio_samples_py..." | tee $LOG_PATH;
maturin build --release >> $LOG_PATH 2>&1;
echo "audio_samples_py built." | tee $LOG_PATH;

echo "Installing audio_samples_py..." | tee $LOG_PATH;
pip install --force-reinstall target/wheels/*.whl >> $LOG_PATH 2>&1;
echo "audio_samples_py installed." | tee $LOG_PATH;