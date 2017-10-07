import argparse
from utils import transform_annotations_file

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Transform annotations file')
    parser.add_argument('--filename', type=str)
    parser.add_argument('--output_filename', type=str, default=None)
    args = parser.parse_args()

    # Process provided file
    print('Start transforming input file: {}'.format(args.filename))
    transform_annotations_file(args.filename, args.output_filename)
    print('Done.')
