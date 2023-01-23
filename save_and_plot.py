import json
import os


def load_epoch_loss(new_data, path):
    if os.path.getsize(path) == 0:
        old_data = {}
    else:
        with open(path, 'r') as f:
            old_data = json.load(f)
    old_data.update(new_data)
    # with open(path, 'r') as f:
    #     file = f.read()
    #     if len(file) <= 0:
    #         old_data = json.load(f)
    #     else:
    #         old_data = {}
    #     old_data.update(new_data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(old_data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    with open('./output/result/LJSpeech\\train_loss.json', 'r') as f:
        file = f.read()
        if len(file) > 0:
            f.close()
            old_data = json.load(f)
    print(old_data)