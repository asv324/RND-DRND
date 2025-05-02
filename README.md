# RND-DRND
Contains all of my code for my 3rd year dissertation project.

## Configuration
The Dockerfile contains all packages necessary to run the code in this repository. Create a Docker Image from this file. Alternatively, you can use a virtual environment, however you will have to convert the Dockerfile into either a .txt for .yaml file containing all relevant packages.

## Atari-Frostbite (RND)

This implementation is based on [jcwleo's  work](https://github.com/jcwleo/random-network-distillation-pytorch). I modified the code and commented each part in order to be comprehensive.

### To train the RND agent
Navigate into the folder and run:
`python train_RND.py`

## Atari-Frostbite (DRND)

This implementation is based on the code written for the paper: "Exploration and Anti-Exploration with Distributional Random Network
Distillation" by Kai Yang, Jian Tao, Jiafei Lyu and Xiu Li. Their code can be found [here](https://github.com/yk7333/DRND)

### To train the DRND agent
Navigate into the folder and run: `python train_DRND.py`

## Point Maze (Both RND and DRND implementations)
### To train the DRND agent
Navigate into the folder and run: `python train_maze.py`

### To train the RND agent
Navigate into the folder and run: `python train_maze_rnd.py`


