#Overview

---

##YOLO (You Only Look Once)

###Goal
Find and locate objects in an image.

###Main idea
- Before YOLO, object detection systems looked at an image **many times**, scanning small parts.
- YOLO is based on convolutional neural networks (CNNs), which are designed to recognize patterns in images.
- YOLO looks at **the whole image once** and predicts both:
	- **What** objects are present.
	- **Where** they are.

###Advantages
- Learns **general patterns**, so it works well on new types of images.
- Super fast: can work in **real time** with cameras (e.g self-driving cars).
	- Base YOLO: ~45 FPS
	- Fast YOLO: ~155 FPS 
- Quite accurate for **clear, well-separated** objects.

###Limitations
- Struggles with **small or overlapping objects**.
- Sometimes bounding boxes are not very precise.

---

##Transformer

###Goal
Understand **sequences of information**, like a text, a speech, or even image patches.

###Main idea
- Uses **attention** to focus on the important parts of the input.
- Looks at the **whole sequence at once**, not just one part at a time.
- In images (like in DETR), the image is split into many small patches, and the Transformer learns how those patches **relate to each other** to understand the objects and their positions.

### Why it's good
- Good at understanding long-range connections.
- Works for text, speech, and images.

---

##DETR (DEtection TRansformer)

###Goal
Detect objects in images, but in a more modern and elegant way.

###Main idea
- Combines a **CNN** to get features with a **Transformer** to understand global context.
- Instead of splitting the image into grids like YOLO, DETR looks at **the whole image** and learns how different sub-parts relate to each other.
- Uses a special **set-based matching system** to make sure each object is detected **only once**.

###Advantages
- Simpler design and easy to implement.
- Understands the global context of the image.

###Limitations
- Takes longer to train, it needs more **computing power**.
- Struggles with **small objects**.

#SUMMARY

YOLO introduced real-time object detection, Transformers brought powerful sequence understanding, and DETR combined both ideas to achieve more context-aware object detection.
