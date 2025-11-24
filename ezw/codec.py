from bitarray import bitarray

from ezw.wave import wavedec_ezw
from ezw.zerotree import ZeroTreeEncoderEZW, ZeroTreeScan, ZeroTreeDecoderEZW
import numpy as np

SOI_MARKER = bytes.fromhex("FFD8")  # Start of Image
SOS_MARKER = bytes.fromhex("FFDA")  # Start of Scan
EOI_MARKER = bytes.fromhex("FFDC")  # End of Image
STUFFED_MARKER = bytes.fromhex("FF00")


# grey image, 2D array
# EZW_passes - number of EZW passes
def encodeEZW(data, w, level, filename, EZW_passes = np.inf):
    # image original size
    shape = data.tensor.shape

    ci = wavedec_ezw(data, w, level)
    enc = ZeroTreeEncoderEZW(ci, w)

    print("Initial Threshold: " + str(enc.start_thresh))
    with open(filename, 'wb') as fh:
        # Write the header
        fh.write(SOI_MARKER)
        fh.write(len(shape).to_bytes(2, 'big'))
        for s in shape:
            fh.write(s.to_bytes(2, 'big'))
        fh.write(int(enc.start_thresh).to_bytes(2, 'big'))
        fh.write(level.to_bytes(2, 'big'))
        i = 0
        writes = float('inf')

        while writes != 0 and i < EZW_passes:
            writes = 0
            fh.write(SOS_MARKER)
            scan = next(iter(enc), None)
            if scan is not None:
                scan.tofile(fh)
                writes += 1
            i += 1

        fh.write(EOI_MARKER)


def decodeEZW(filename, wavelet):
    with open(filename, 'rb') as fh:
        soi = fh.read(2)
        if soi != SOI_MARKER:
            raise Exception("Start of Image marker not found!")

        shape_len = int.from_bytes(fh.read(2), 'big')
        shape = np.zeros(shape_len, dtype=np.int64)
        for i in range(shape_len):
            shape[i] = int.from_bytes(fh.read(2), 'big')

        threshold = int.from_bytes(fh.read(2), 'big')
        level = int.from_bytes(fh.read(2), 'big')
        decoder = ZeroTreeDecoderEZW(shape, threshold, wavelet, level)

        cursor = fh.read(2)
        if cursor != SOS_MARKER:
            raise Exception("Scan's not found!")

        isDominant = True
        while cursor != EOI_MARKER:
            buffer = bytes()
            while len(buffer) < 2 or ((buffer[-2:] != EOI_MARKER) and (buffer[-2:] != SOS_MARKER)):
                buffer += fh.read(1)

            buffer, cursor = buffer[:-2], buffer[-2:]
            buffer = buffer.replace(STUFFED_MARKER, b'\xff')

            ba = bitarray()
            ba.frombytes(buffer)

            if len(ba) != 0:
                scan = ZeroTreeScan.from_bits(ba, isDominant)
                decoder.process(scan)

            isDominant = not isDominant

        image = np.zeros(shape)
        image = decoder.getImage()
        #image.tensor = image.tensor.astype(np.uint8)

    return image