import argparse
import gzip
import io
import os
import subprocess
from collections import Counter

import progressbar
from clearml import Task


def convert_text(idx, input_txt, output_dir):

    data_lower = os.path.join(output_dir, "{}_lower.txt.gz".format(idx))

    print("\nConverting to lowercase ...")
    with io.TextIOWrapper(
        io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
    ) as file_out:

        # Open the input file either from input.txt or input.txt.gz
        _, file_extension = os.path.splitext(input_txt)
        if file_extension == ".gz":
            file_in = io.TextIOWrapper(
                io.BufferedReader(gzip.open(input_txt)), encoding="utf-8"
            )
        else:
            file_in = open(input_txt, encoding="utf-8")

        for line in progressbar.progressbar(file_in):
            line_lower = line.lower()
            file_out.write(line_lower)

        file_in.close()

    return data_lower

def build_intermediate_lm(args, idx, data_lower):
    print("\nCreating ARPA file ...")
    lm_path = os.path.join(args.output_dir, "{}_lm.arpa".format(idx))
    intermediate_lm_path = os.path.join(args.output_dir, "{}_lm".format(idx))
    subargs = [
        os.path.join(args.kenlm_bins, "lmplz"),
        "--order",
        str(args.arpa_order),
        "--temp_prefix",
        args.output_dir,
        "--memory",
        args.max_arpa_memory,
        "--text",
        data_lower,
        "--arpa",
        lm_path,
        "--prune",
        *args.arpa_prune.split("|"),
        "--intermediate",
        intermediate_lm_path,
    ]

    # if args.discount_fallback:
    #     subargs += ["--discount_fallback"]

    subprocess.check_call(subargs)

    return intermediate_lm_path

def binarize_lm(args, interpolated_lm_arpa):

    # Quantize and produce trie binary.
    print("\nBuilding interpolated_lm.binary ...")
    binary_path = os.path.join(args.output_dir, "interpolated_lm.binary")
    subprocess.check_call(
        [
            os.path.join(args.kenlm_bins, "build_binary"),
            "-a",
            str(args.binary_a_bits),
            "-q",
            str(args.binary_q_bits),
            "-v",
            args.binary_type,
            interpolated_lm_arpa,
            binary_path,
        ]
    )
    print("\nBinary file written to {}".format(binary_path))

def main():
    parser = argparse.ArgumentParser(
        description="Generate an interpolated language model from multiple corpora."
    )
    parser.add_argument(
        "--name",
        help="Name to use for writing interpolated LM files",
        type=str,
        required=True,
        nargs=1
    )
    parser.add_argument(
        "--output_dir", help="Directory path for the output", type=str, required=True
    )
    parser.add_argument(
        "--input_txts",
        help="Paths to file.txt or file.txt.gz with sample sentences",
        type=str,
        required=True,
        nargs="+"
    )
    parser.add_argument(
        "--weights",
        help="Weights to use for input LMs when interpolating",
        type=str,
        required=True,
        nargs="+"
    )
    parser.add_argument(
        "--kenlm_bins",
        help="File path to the KENLM binaries lmplz, filter and build_binary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_arpa_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--binary_a_bits",
        help="Build binary quantization value a in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_q_bits",
        help="Build binary quantization value q in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_type",
        help="Build binary data structure type",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    assert len(args.input_txts) == len(args.weights)

    intermediate_lms = []

    for idx, vals in enumerate(zip(args.input_txts, args.weights)):
        input_txt, weight = vals
        data_lower = convert_text(idx, input_txt, args.output_dir)
        intermediate_lm_path = build_intermediate_lm(args, idx, data_lower)
        intermediate_lms.append(intermediate_lm_path)

    interpolated_lm_arpa = os.path.join(args.output_dir, "{}.arpa".format(args.name[0]))

    subargs = [
        os.path.join(args.kenlm_bins, "interpolate"),
        "-m",
        " ".join(intermediate_lms),
        "-w",
        " ".join(args.weights),
    ]
 
    interpolate_cmd = "{} -m {} -w {}".format(
        os.path.join(args.kenlm_bins, "interpolate"),
        " ".join(intermediate_lms),
        " ".join(args.weights)
        )

    print(interpolate_cmd)
    with open(interpolated_lm_arpa, "w") as f:
        #subprocess.check_call(subargs, shell=True,  stdout=f)
        subprocess.check_call(interpolate_cmd, shell=True, stdout=f)

    # will skip arpa file filtering for now
    # need some logic to include all ngrams from customer data
    # and filter out ngrams past some threshold for other data

    binarize_lm(args, interpolated_lm_arpa)

if __name__ == "__main__":
    main()
