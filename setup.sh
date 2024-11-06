git clone https://github.com/vinhpad/ALTOP-ENHANCE.git
git checkout feature/docred
cd ALTOP-ENHANCE/dataset/docred
wget https://drive.google.com/file/d/1AHUm1-_V9GCtGuDcc8XrMUCJE8B-HHoL/view?usp=drive_link
wget https://drive.google.com/file/d/1lAVDcD94Sigx7gR3jTfStI66o86cflum/view?usp=drive_link
wget https://drive.google.com/file/d/1NN33RzyETbanw4Dg2sRrhckhWpzuBQS9/view?usp=drive_link
wget https://drive.google.com/file/d/1Qr4Jct2IJ9BVI86_mCk_Pz0J32ww9dYw/view?usp=drive_link
cd ../../
pip install -r requirement.txt
bash scripts/run_docred.sh