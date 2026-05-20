import numpy as np


_SEQ_LUT = np.zeros((256, 4), dtype=np.float32)
_SEQ_LUT[ord('A')] = [1.0, 0.0, 0.0, 0.0]
_SEQ_LUT[ord('C')] = [0.0, 1.0, 0.0, 0.0]
_SEQ_LUT[ord('G')] = [0.0, 0.0, 1.0, 0.0]
_SEQ_LUT[ord('T')] = [0.0, 0.0, 0.0, 1.0]
_SEQ_LUT[ord('a')] = [1.0, 0.0, 0.0, 0.0]
_SEQ_LUT[ord('c')] = [0.0, 1.0, 0.0, 0.0]
_SEQ_LUT[ord('g')] = [0.0, 0.0, 1.0, 0.0]
_SEQ_LUT[ord('t')] = [0.0, 0.0, 0.0, 1.0]


def sequence_encode(seq):
    # Vectorized char->onehot encoding via ASCII LUT (N/X/others map to zeros).
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _SEQ_LUT[seq_bytes]
