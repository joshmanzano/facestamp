rm -rf logs/*
rm profiling_stats
python3 train.py
python3 read_profile.py