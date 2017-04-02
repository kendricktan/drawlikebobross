#!/bin/bash

# Make dictionary to store images
mkdir -p images

# Painting URLS
url="http://www.twoinchbrush.com/images/painting"
ext=".png"

# Iterate through all 411 images
for i in `seq 1 411`;
do
    c_url=$url$i$ext
    wget $c_url -O "images/"$i$ext &
done

# Wait until the background jobs are done
for job in `jobs -p`
do
    wait $job
done

# Clean up corrupted files
find images/. -size 0c -delete
echo "=========> !! Done !! <========="
