*Overview

**YOLO (You Only Look Once)

***Goal: Find and locate objects in an image.

***Main idea
- Before YOLO, object detection systems looked at an image many times, scanning small parts.
- YOLO is based on convolutional neural networks (CNNs), which are designed to recognize patterns in images.
- YOLO looks at the whole image once and predicts both:
- What objects are present.
- Where they are (the bounding boxes).

***Advantages
- Super fast , can work in real time (great for cameras or self-driving cars).
- Quite accurate for clear, well-separated objects.

***Limitations
- Struggles with small or overlapping objects.

**Transformer

***Goal: Understand sequences of information, like a text, a speech, or even image patches.

***Main idea
- Transformers were first created for language tasks.
- They use an attention mechanism, which lets the model focus on the most relevant parts of the input.
- In images (like in DETR), the image is split into many small patches, and the Transformer learns how those patches relate to each other to understand the objects and their positions.

**DETR (DEtection TRansformer)

***Goal: Detect objects in images, but in a more modern and elegant way.

***Main idea
- DETR uses a Transformer to handle object detection.
- DETR can be seen as a next-generation object detector that replaces YOLO’s CNN-based approach with a Transformer-based one.
- Instead of splitting the image into grids like YOLO, DETR looks at the whole image and learns how different sub-parts relate to each other.

***Advantages
- Simpler design.
- Understands the global context of the image.
- More flexible, it can adapt to different kinds of scenes.

***Limitations
- Takes longer to train, it needs more computing power.

*SUMMARY

YOLO introduced real-time object detection, Transformers brought powerful sequence understanding, and DETR combined both ideas to achieve more context-aware object detection.
