#!/bin/bash

# URL of the file to download
URL="https://github.com/summmeer/session-based-news-recommendation/files/9467559/articles_timeDict_103630.pkl.gz"
# Name of the file to save after download
FILENAME="articles_timeDict_103630.pkl.gz"
# Destination directory
DEST_DIR="data"

# Download the file using curl
curl -L $URL -o $FILENAME

# Check if the download was successful
if [ -f "$FILENAME" ]; then
    # Unzip the file
    gunzip -f $FILENAME
    echo "File downloaded and unzipped successfully."
    # Check if the data directory exists
    if [ ! -d "$DEST_DIR" ]; then
        # Create the directory if it doesn't exist
        mkdir "$DEST_DIR"
    fi
    # Move the file to the data directory
    mv "${FILENAME%.gz}" "$DEST_DIR/"
    echo "File moved to $DEST_DIR"
else
    echo "Failed to download the file."
fi