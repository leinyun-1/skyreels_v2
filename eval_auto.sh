source /opt/conda/etc/profile.d/conda.sh
conda activate /dockerdata/venv/causvid

cd experiments
python libs/test_transformers.py

cd ../skyreels_v2
sh eval.sh 

cd ../experiments
python libs/test_imageio.py

cd ../DiffSynth-Studio
conda activate diff
python test.py 


