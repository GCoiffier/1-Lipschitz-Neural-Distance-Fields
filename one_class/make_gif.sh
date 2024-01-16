for f in *.png
do
convert $f -background white -alpha remove -alpha off ${f%.png}.jpg
done
convert `ls render_*.jpg` -coalesce -layers optimize render.gif
rm render_*.jpg