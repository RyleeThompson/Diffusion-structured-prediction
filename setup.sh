source /coreflow/venv/bin/activate
/coreflow/venv/bin/python3 -m pip install --upgrade pip

pip install -r utils/cluster/requirements.txt
# pip install numpy --upgrade
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
source utils/cluster/setup_blobby.sh
pip install opencv-python==4.4.0.40

pip freeze

aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300 s3 cp s3://coco_predictions/faster_rcnn_X_101_32x8d_FPN_3x.tar.gz dataset/
aws --endpoint-url https://blob.mr3.simcloud.apple.com --cli-read-timeout 300 s3 cp s3://coco_predictions/coco-data.h5 dataset/
tar -xf dataset/faster_rcnn_X_101_32x8d_FPN_3x.tar.gz -C dataset/
mv dataset/faster_rcnn_X_101_32x8d_FPN_3x_temp/ dataset/faster_rcnn_X_101_32x8d_FPN_3x

trove mount dataset/coco2017@1.1.0 dataset/
echo 'set up complete'
