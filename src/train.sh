
export PYTHONPATH="${PYTHONPATH}:/home/lingling/code/DiffmotionEmotionGesture/src/pymo"
CUDA_VISIBLE_DEVICES=1 nohup python train_gesture_generation.py >train_beat_0629_1.txt &
# CUDA_VISIBLE_DEVICES=1  python train_gesture_generation.py >train_beat_0629_1.txt 