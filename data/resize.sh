# Use ./resize.sh path/
# to resize all jpegs in path/

images=$(ls $1 | grep -c ".jpeg")
count=1

echo -ne "Started processing $images images.\n"

for name in $1*.jpeg; do
    percent=$(echo "scale = 3; $count*100/$images" | bc)
    echo -ne "$percent% complete.\r"
    convert -resize 256x256\! $name $name
    ./contour $name
    let "count++"
done

echo -ne "\n"
