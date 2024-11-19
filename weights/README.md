# Weight converter

Convert the older pytorch models to the newer one, supported by candle.

## Usage

1. Install dependency (only `torch` is needed).
1. Move to `input` and Run `download-efficientnet.sh` (without advprop) or `download-efficientnetadv.sh` (with advprop).
1. Return to `weights` and run `python -m weights`.
1. Copy the files in the `output` folder to `~/.cache/efficientnet-burn/`.
