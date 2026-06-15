"""Go substrate corpus = the Go standard library — dep-free, interface-rich, well-tested:
the cleanest large Go corpus for the CHA/RTA cleanness dial. Regenerate the TSVs with the
Go extractor (Go installed user-space at ~/go-sdk):

    cd artifacts_scripts/curriculum/gap_recover/adapters/go/extractor
    PATH=~/go-sdk/bin:$PATH GOPATH=~/go go run . -out <repo>/cache/code_gap/go/std -pkgs std
"""
PATTERNS = ["std"]
