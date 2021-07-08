import os
source_file = 'E:\Code\PyCharm\Video\Data\data_thchs30'
def source_get(source_file):
    train_file = source_file + '/data'
    label_lst = []
    wav_lst = []
    for root, dirs, files in os.walk(train_file):
        for file in files:
            if file.endswith('.wav') or file.endswith('.WAV'):
                wav_file = os.sep.join([root, file])
                label_file = wav_file + '.trn'
                wav_lst.append(wav_file)
                label_lst.append(label_file)

    return label_lst, wav_lst


label_lst, wav_lst = source_get(source_file)

print(label_lst[:10])
print(wav_lst[:10])
