cd $1;
source .env/bin/activate;
pip install maturin;
maturin build --sdist;