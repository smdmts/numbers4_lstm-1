import os
import numpy as np

# root path
root = 'data'
cnt  = 0

def get_numbers4(type, ratio):
    """ Gets previous winning number from the archive.

    This function returns the training, validation and test sets,
    eatch of which is represented as a long array of IDs. 

    """
    datasize = len(open(os.path.join(root, 'numbers4.csv')).readlines())
    print('\nDatasize : {}'.format(datasize))
    learn = int(datasize * ratio)
    other = int(datasize * ((1 - ratio) / 2)) + cnt
    print(' * train : {}'.format(learn))
    print(' * valid : {}'.format(other))
    print(' * test  : {}'.format(other))

    train = _retrieve_numbers('train_{}'.format(type), type, size=int(learn))
    valid = _retrieve_numbers('valid_{}'.format(type), type, size=int(other))
    test  = _retrieve_numbers('test_{}'.format(type),  type, size=int(other)+cnt)
    
    return train, valid, test

def get_vocab(type):
    """Gets numbers vocabulary.

    Returns:
        dict: Dictionary that maps words to corresponding word IDs. The IDs are
        used in the winning numbers datasets.

    .. seealso::
       See :func:`get_numbers4` for the actual datasets.

    """
    return _retrieve_word_vocab(type, None)

def create_or_load(path, creator, loader):
    if os.path.exists(path):
        return loader(path)

    return creator(path)

def _retrieve_numbers(name, type, size):
    def creator(path):
        vocab = _retrieve_word_vocab(type=type, size=size)
        words = _load_words(type=type, size=size)
        x = np.empty(len(words), dtype=np.int32)
        for i, word in enumerate(words): 
            x[i] = vocab[word]
        np.savez_compressed(path, x=x)
        return {'x': x}

    path = os.path.join(root, name)
    loaded = create_or_load(path, creator, np.load)
    return loaded['x']

def _retrieve_word_vocab(type, size):
    """ Create vocab from loaded csv """
    def creator(path):
        vocab = {}
        index = 0
        if type == 'n4_one':
            words = _load_words(type=type, size=size)
            with open(path, 'w') as f:
                for word in words:
                    if word not in vocab:
                        vocab[word] = index
                        index += 1
                        f.write(word + '\n')
        elif type == 'n4':
            with open(path, 'w') as f:
                for i in range(0, 10000):
                    word = '{:04d}\n'.format(i)
                    vocab[word] = index
                    index += 1
                    f.write(word)
        return vocab

    def loader(path):
        vocab = {}
        with open(path) as f:
            for i, word in enumerate(f):
                vocab[word.strip()] = i
        return vocab

    path = os.path.join(root, 'vocab_{}.txt'.format(type))
    return create_or_load(path, creator, loader)

def _load_words(type, size):
    """ Create words from csv.
    All numbers are concatenated by End-of-Numbers mark '<eos>',
    which is treated as one of the vocabulary.
    type : default is 'n4'
        dtype : str
        can choose in ['n4' or 'n4_one']
        if type is 'n4', the model is trained by 4 characters.
        'n4_one' type, it is trained by one.

    Returns : 
        list : numbers data convert to words
    """
    global cnt

    words = []
    # load csv file
    with open(os.path.join(root,'numbers4.csv'),'r') as csv:
        for line in csv.readlines()[cnt:cnt+size]:
            if type == 'n4':
                # 1234, 5678, ...
                str_line = line.strip()
                words.append(str_line[0] + str_line[2] + str_line[4] + str_line[6])
            elif type == 'n4_one':
                # '1', '2', '3', '4', '<eos>', ...
                words += line.strip().split(',')
                words.append('<eos>')
    
    return words

if __name__ == '__main__':
    a, b ,c = get_numbers4(type='n4', ratio=1/3)
    print(a.shape[0])











