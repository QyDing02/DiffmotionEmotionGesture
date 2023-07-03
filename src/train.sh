
export PYTHONPATH="${PYTHONPATH}:/home/lingling/code/DiffmotionEmotionGesture/src/pymo"
# CUDA_VISIBLE_DEVICES=0 nohup python train_gesture_generation.py >../train_beat_test.txt &
# CUDA_VISIBLE_DEVICES=1  python train_gesture_generation.py >train_beat_0629_test_addEmotion.txt 
CUDA_VISIBLE_DEVICES=1  python train_gesture_generation.py