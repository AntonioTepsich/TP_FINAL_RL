import os
import shutil

SRC_DIR = "experiments"
DST_DIR = "runs"

os.makedirs(DST_DIR, exist_ok=True)

for exp_name in os.listdir(SRC_DIR):
    exp_path = os.path.join(SRC_DIR, exp_name)
    tensorboard_path = os.path.join(exp_path, "tensorboard")
    if os.path.isdir(tensorboard_path):
        dst_exp_path = os.path.join(DST_DIR, exp_name)
        os.makedirs(dst_exp_path, exist_ok=True)
        for file in os.listdir(tensorboard_path):
            src_file = os.path.join(tensorboard_path, file)
            dst_file = os.path.join(dst_exp_path, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
        print(f"Copiados archivos de {exp_name}")
print("Listo.")