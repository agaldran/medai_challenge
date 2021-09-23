mkdir data
mkdir data/instruments
mkdir data/polyps
wget https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip
unzip Kvasir-SEG.zip
rm Kvasir-SEG.zip
mv Kvasir-SEG/images data/polyps/
mv Kvasir-SEG/masks data/polyps/
rm -r Kvasir-SEG

wget https://datasets.simula.no/kvasir-instrument/kvasir-instrument.tar.gz
tar -xf kvasir-instrument.tar.gz
tar -xf kvasir-instrument/images.tar.gz
mv images data/instruments
tar -xf kvasir-instrument/masks.tar.gz
mv masks data/instruments
rm kvasir-instrument.tar.gz
rm -r kvasir-instrument/

python prepare_data.py

rm -r data/polyps/images
rm -r data/polyps/masks
rm -r data/instruments/images
rm -r data/instruments/masks

