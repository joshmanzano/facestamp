import utils
import argparse
import model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify file to analyze')
    parser.add_argument('filename')

    args = parser.parse_args()

    filename = args.filename

    secret, region = utils.get_secret_string(filename)

    print(secret)

