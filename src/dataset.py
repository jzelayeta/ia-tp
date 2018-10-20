import numpy as np



def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


X_train = np.genfromtxt(r"/Users/julianzelayeta/code/ia-tp/files/1981-2010.csv",delimiter=" , ", dtype=None, skip_header = 1)
chunks = chunkIt(X_train, 76)
print(chunks[1])


