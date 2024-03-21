import os



for seed in [1,2,3,4]:
    
    screen_log_path = "/g/stegle/ueltzhoe/bicycle/notebooks/experiments/heart_organoids/screen_log_bicycle_seed%d.txt" % seed
    
    cmd = "screen -L -Logfile %s -dmS bicycle_run_%d srun -p gpu-el8 -N 1 -C gpu=3090 --gpus 4 --cpus-per-gpu 8 --mem-per-gpu 64G --pty --time=28:00:00 -E $SHELL /g/stegle/ueltzhoe/bicycle/notebooks/experiments/heart_organoids/run_heart_organoids.sh %d" % (screen_log_path, seed, seed)
    print('Starting: ' + cmd)
    os.system(cmd)