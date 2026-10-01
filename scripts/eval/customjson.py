"""CustomJSON loads user-provided conversations but has no evaluation scorer."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.error("CustomJSON loads user-provided conversations but has no evaluation scorer.")


if __name__ == "__main__":
    main()

