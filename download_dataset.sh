#!/bin/bash

# Directory where the files will be saved
DEST_DIR="./data/amazon"

# Create the destination directory if it doesn't exist
mkdir -p $DEST_DIR

# List of URLs to download
URLS=(
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Video_Games.json.gz"
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz"
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Luxury_Beauty.json.gz"
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Luxury_Beauty.json.gz"
)

# Loop through each URL
for URL in "${URLS[@]}"; do
    # Extract the file name from the URL
    FILE_NAME=$(basename $URL)
    
    # Full path where the file will be saved
    FILE_PATH="$DEST_DIR/$FILE_NAME"
    
    # Download the file to the destination directory
    echo "Downloading $FILE_NAME..."
    curl -o $FILE_PATH $URL
    
    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful: $FILE_NAME"
        
        # Unzip the file in the destination directory
        echo "Unzipping $FILE_NAME..."
        gunzip $FILE_PATH
        
        # Check if the unzipping was successful
        if [ $? -eq 0 ]; then
            echo "Unzipping successful: ${FILE_NAME%.gz}"
        else
            echo "Unzipping failed for $FILE_NAME"
        fi
    else
        echo "Download failed for $FILE_NAME"
    fi
done