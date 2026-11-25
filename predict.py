"""Command line: classify one or more photos and say which bin each one goes in.

    python predict.py photo1.jpg photo2.jpg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from binsort import BinRules, WasteClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--bins", default=None, help="path to a bins.json (defaults to the one in the repo)")
    args = ap.parse_args()
    clf, rules = WasteClassifier(), BinRules(args.bins)
    for path in args.images:
        r = clf.predict(path)
        b = rules.lookup(r["label"])
        top = ", ".join(f"{t['label']} {t['score']:.2f}" for t in r["top"])
        print(f"{path}\n  class : {r['label']}\n  bin   : {b['bin']}\n  tip   : {b['tip']}\n  scores: {top}")


if __name__ == "__main__":
    main()
