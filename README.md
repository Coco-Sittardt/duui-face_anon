https://github.com/hanweikung/face_anon_simple/tree/main

## what can it do?

- anonymization with a diferent face
- redaction with black, pixel or blur
- anonymization of multiple faces in an image
- swapping face in image A with face from image B 
-     # options: single_align, multiple_align, combine, swap, redact

## Citations 
```bibtex
@InProceedings{Kung_2025_WACV,
    author    = {Kung, Han-Wei and Varanka, Tuomas and Saha, Sanjay and Sim, Terence and Sebe, Nicu},
    title     = {Face Anonymization Made Simple},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {1040-1050}
}
```


```python
# the base selection between which anonymization is run
anon_type = request.anon_type
# the amount of anonymization
anon_degree = request.anon_degree
# set if the anon_type is redaction, then can choose again between blur, black or pixel
redact_type = request.redact_type
blur = request.blur
pixel = request.pixel
diffusion_model = request.diffusion_model
clip_model = request.clip_model
seed = request.seed
guidance = request.guidance
inference_steps = request.inference_steps
vis_input = request.vis_input
height = request.height
width = request.width
```

- ERROR: git+https://github.com/hanweikung/face_anon_simple.git does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
