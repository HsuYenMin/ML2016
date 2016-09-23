from PIL import Image
import sys
filename = sys.argv[1]
im = Image.open(filename)
Output = Image.new("RGB",im.size,"white")
Output = im.rotate(180)
Output.save("ans2.png")
