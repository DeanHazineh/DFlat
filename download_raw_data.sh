#!/bin/bash

wget -O dflat/metasurface/data.zip 'https://www.dropbox.com/scl/fi/efzz37tlejkkplo7pe7vs/data.zip?rlkey=malv67btexvfhkyhbiasgrai0&dl=1'

cd dflat/metasurface/
echo "Attempting to unzip the file..."

unzip -o data.zip
if [ $? -eq 0 ]; then
    echo "Unzipping completed successfully."
else
    echo "Failed to unzip the file."
fi

rm data.zip
