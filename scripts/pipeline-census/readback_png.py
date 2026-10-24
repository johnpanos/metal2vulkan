#!/usr/bin/env python3
"""Write a command-replay readback (DIR/<id>.bin, rows packed) of an 8-bit four-channel texture
as a PNG, to compare a replayed frame with the device's screenshot.

    readback_png.py BIN WIDTH HEIGHT OUT.png [--order bgra|rgba]"""
import argparse
import struct
import zlib

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('bin')
ap.add_argument('width', type=int)
ap.add_argument('height', type=int)
ap.add_argument('out')
ap.add_argument('--order', choices=('bgra', 'rgba'), default='bgra')
a = ap.parse_args()
data = open(a.bin, 'rb').read()
row = a.width * 4
if len(data) < row * a.height:
    raise SystemExit('%s: %d bytes, %dx%d needs %d' % (a.bin, len(data), a.width, a.height, row * a.height))
raw = bytearray()
for y in range(a.height):
    line = bytearray(data[y * row:(y + 1) * row])
    if a.order == 'bgra':
        line[0::4], line[2::4] = line[2::4], line[0::4]
    raw += b'\0' + line
chunk = lambda kind, body: struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', a.width, a.height, 8, 6, 0, 0, 0)) + \
    chunk(b'IDAT', zlib.compress(bytes(raw), 6)) + chunk(b'IEND', b'')
open(a.out, 'wb').write(png)
print(a.out)
