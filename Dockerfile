# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN pip3 install atari-py
RUN pip3 install gym==0.23.1
RUN pip3 install gym[atari]
RUN pip3 install gym[accept-rom-license]
RUN pip3 install tensorboard==2.18.0
RUN pip3 install jax==0.4.13
RUN pip3 install jax-jumpy==1.0.0
RUN pip3 install jaxlib==0.4.13
RUN pip3 install jaxopt==0.6
RUN pip3 install nes-py==8.2.1
RUN pip3 install gym-super-mario-bros
RUN pip3 install opencv-python
RUN pip3 install protobuf==3.20.0
RUN pip3 install mujoco
RUN pip3 install mujoco-py
RUN pip3 install pyrallis
RUN pip3 install optax
RUN pip3 install flax
RUN pip3 install distrax
RUN pip3 install cython
RUN pip3 install pynvml
RUN pip3 install gin-config
# offline RL
RUN pip3 install minari
# cv2 dependencies
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y libgl1
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libglib2.0-0
RUN pip3 install numpy==1.25.0
RUN pip3 install gymnasium-robotics
RUN pip3 install seaborn