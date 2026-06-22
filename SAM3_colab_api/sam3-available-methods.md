# Code

```py
from sam3.model.sam3_image_processor import Sam3Processor

# Look at the class blueprint directly without loading any weights!
methods = [m for m in dir(Sam3Processor) if not m.startswith('_')]
print("AVAILABLE METHODS:")
for m in methods:
    print(f"- {m}")
```

```py
from sam3.model.sam3_image_processor import Sam3Processor
help(Sam3Processor.add_geometric_prompt)
help(Sam3Processor.reset_all_prompts)
help(Sam3Processor.set_confidence_threshold)
help(Sam3Processor.set_image)
help(Sam3Processor.set_image_batch)
help(Sam3Processor.set_text_prompt)
```

# Result

```text
AVAILABLE METHODS:
- add_geometric_prompt
- reset_all_prompts
- set_confidence_threshold
- set_image
- set_image_batch
- set_text_prompt
```

```text
Help on function add_geometric_prompt in module sam3.model.sam3_image_processor:

add_geometric_prompt(self, box: List, label: bool, state: Dict)
    Adds a box prompt and run the inference.
    The image needs to be set, but not necessarily the text prompt.
    The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
    The label is True for a positive box, False for a negative box.

Help on function reset_all_prompts in module sam3.model.sam3_image_processor:

reset_all_prompts(self, state: Dict)
    Removes all the prompts and results

Help on function set_confidence_threshold in module sam3.model.sam3_image_processor:

set_confidence_threshold(self, threshold: float, state=None)
    Sets the confidence threshold for the masks

Help on function set_image in module sam3.model.sam3_image_processor:

set_image(self, image, state=None)
    Sets the image on which we want to do predictions.

Help on function set_image_batch in module sam3.model.sam3_image_processor:

set_image_batch(self, images: List[numpy.ndarray], state=None)
    Sets the image batch on which we want to do predictions.

Help on function set_text_prompt in module sam3.model.sam3_image_processor:

set_text_prompt(self, prompt: str, state: Dict)
    Sets the text prompt and run the inference
```