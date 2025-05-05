# cartilage_seg

## Environment setup
Please note that this repository has only been tested only on a Windows PC.

Clone the repository and machine learning model using the following command
```
git clone https://github.com/alvinkimbowa/clarius_model_integration.git
```

Install the necessary dependents, create a python environment and install the requirements using the commands below. Please note that this repository has only been tested with python environments.

```
python -m venv venv

cd venv/Scripts
activate
cd ../../

pip install -r requirements.txt
```

# Run Inference
Run the `inference_engine.py` file, adjusting the data paths accordingly.
```
python inference_engine.py
```

# Notes
- The current model was trained on 1787 images obtained with the GE LOGIQ P9 R3 ultrasound system with the L312-RS wideband linear array probe.
However, the model can generalize to other devices, thanks to the contrast and intensity invariant Mono2D layer I integrated ([See paper](https://arxiv.org/abs/2503.09050)).

- The model achieves Dice of **95.56%** on test images from GE LOGIQ P9 R3 images, and an average of **94.27%** when evaluated with images from GE LOGIQe and Clarius L15 HD3.


### ✌ Now every knee 🦿 you set your eyes upon... shall be imaged 😎
