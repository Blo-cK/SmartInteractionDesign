from ultralytics import YOLOE

#Use this for prompted test

model = YOLOE("yoloe-11l-seg.pt")
names = ["shoes","backpack","shorts","boots","shorts","pants","hat","shirt","glasses"]
model.set_classes(names, model.get_text_pe(names))
#set test video here
model.predict("testvidlowfps.mp4", show=True, save=True)

#use this for unprompted test

# model = YOLOE("yoloe-11l-seg-pf.pt")
## Set video here
# results = model.predict("testvidlowfps.mp4",
#                         show=True,
#                         save=True,
#                         show_conf=False)

