import flatbuffers
import argparse

parser = argparse.ArgumentParser(description='Check model inputs')
parser.add_argument('file', type=str, help='File path')
file = parser.parse_args().file

buf = open(file, 'rb').read()

print(buf)
