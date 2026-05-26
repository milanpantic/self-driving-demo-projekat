import os
import shutil
import kagglehub

def main():
    # Download dataset (u cache)
    path = kagglehub.dataset_download(
        "andy8744/udacity-self-driving-car-behavioural-cloning"
    )

    print(f"[INFO] Downloaded to: {path}")

    # Gde hoćemo na host-u (pošto je /workspace mount)
    target_dir = "/workspace/complete_dataset"

    # Ako već postoji → obriši (opciono)
    #if os.path.exists(target_dir):
     #   print("[INFO] Removing existing dataset...")
     #   shutil.rmtree(target_dir)

    # Kopiranje
    print("[INFO] Copying dataset to /workspace...")
    shutil.copytree(path, target_dir)

    print(f"[DONE] Dataset available at: {target_dir}")


if __name__ == "__main__":
    main()