# Traffic Sign Recognition Using CNN
This is a university assignment where students were asked to build an AI model in order to recognize the proposed objects.
## Setup Guide
Once you clone the repo, the next thing you should do is to download images dataset from GTSRB or German Traffic Sign Recognition Benchmark.

Download link: [GTSRB Dataset Archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)

Create a "src" folder and explicitly extract the desired images to the src folder.

## Rename filenames
Before continuing to copy other images, you should rename the existing files following the given format: "XX_"

The first 2 characters is the label the data type, followed by an underscore to indicate the images are yet to be renamed by the python script file.

Run
```bash
python
>>> from common import files_rename
>>> files_rename()
```

The function will begin to rename files with random assigned numbers followed by the data type separated by an underscore.

# Voila!
You may start to train your model in main.ipynb!
