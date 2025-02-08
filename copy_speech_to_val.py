import os

if __name__ == "__main__":
    with open("./filelists/ljs_audio_text_test_filelist_1300.txt", encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split("|") for line in f]
    print(len(filepaths_and_text))

    with open("./preprocessed_data/LJSpeech/train.txt", encoding='utf-8') as f:
        train_filepaths_and_text = [line.strip().split("|") for line in f]
    with open("./preprocessed_data/LJSpeech/val.txt", encoding='utf-8') as f:
        val_filepaths_and_text = [line.strip().split("|") for line in f]
    with open("./preprocessed_data/LJSpeech/test.txt", encoding='utf-8') as f:
        test_filepaths_and_text = [line.strip().split("|") for line in f]

    out = []
    all_data = train_filepaths_and_text+val_filepaths_and_text+test_filepaths_and_text
    for i, basename in enumerate([line[0] for line in all_data]):
        for audio_path, text in filepaths_and_text:
            audio_filename = audio_path.strip().split('/')[-1].split('.')[0]
            if audio_filename == basename:
                out.append("|".join([all_data[i][0], all_data[i][1], all_data[i][2], all_data[i][3]]))
                print(i, basename)
    with open(os.path.join("./preprocessed_data/LJSpeech/", "test_1300.txt"), "w", encoding="utf-8") as f:
        for m in out:
            f.write(m + "\n")