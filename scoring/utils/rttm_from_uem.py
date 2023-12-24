#!/usr/bin/env python
"""Split RTTM into multiple files."""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from collections import namedtuple
import itertools
import os
import sys

PY2 = sys.version_info[0] == 2

if PY2:
    FileExistsError = OSError
else:
    from builtins import FileExistsError


Turn = namedtuple('Turn', ['type', 'fid', 'channel', 'onset', 'dur', 'ortho', 'speaker_type', 'speaker_id', 'score', 'temp'])

def make_dir(dirpath):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

def load_rttm(fn):
    """Load turns from RTTM file."""
    with open(fn, 'rb') as f:
        turns = []
        for line in f:
            fields = line.decode('utf-8').strip().split()
            turns.append(Turn(*fields))
    return turns

def load_uem(fn):
    """Load list files from UEM file."""
    with open(fn, 'rb') as f:
        fileID = []
        for line in f:
            fields = line.decode('utf-8').strip().split()
            fileID.append(fields[0])
    return fileID

def write_rttm(fn, turns):
    """Write turns to RTTM file."""
    with open(fn, 'wb') as f:
        turns = sorted(
            turns, key=lambda x: (x.fid, float(x.onset), float(x.dur)))
        for turn in turns:
            line = ' '.join(turn)
            f.write(line.encode('utf-8'))
            f.write(b'\n')


# def groupby(iterable, keyfunc):
#     """Wrapper around ``itertools.groupby`` which sorts data first."""
#     iterable = sorted(iterable, key=keyfunc)
#     for key, group in itertools.groupby(iterable, keyfunc):
#         yield key, list(group)


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description='extect RTTM file based on UEM.', add_help=True)
    parser.add_argument(
        'src_rttm_fn', metavar='rttm', help='RTTM file to split')
    parser.add_argument(
        'uem_fn', metavar='uem', help='UEM file')
    parser.add_argument(
        'output_dir', help='output directory for new RTTM files')
    parser.add_argument(
        'type', metavar='type', help='ref or sys')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    make_dir(args.output_dir)
    # print(f"src_rttm_fn: {args.src_rttm_fn}")
    turns = load_rttm(args.src_rttm_fn)
    # print(f"turns: {turns}")
    file_list = load_uem(args.uem_fn)
    domain=args.uem_fn.split('/')[-1].split('.')[0]
    
    extracted_turns = []
    for turn in turns:
        if turn.fid in file_list:
            extracted_turns.append(turn)
    # for turn in extracted_turns:
    #     print(f"turn: {turn}")
    dest_rttm_fn = os.path.join(args.output_dir, args.type + '_' + domain + '.rttm')
    # print(f"dest_rttm_fn: {dest_rttm_fn}")
    write_rttm(dest_rttm_fn, extracted_turns)

if __name__ == '__main__':
    main()
