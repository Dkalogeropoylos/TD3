Installation for windows

```
python -m venv env

env\Scripts\activate

pip install -r requirements.txt
```


Installation for ubuntu

```
python3 -m venv env

source env/bin/activate

pip install -r requirements.txt
```

The experiment has 2 main modes
1. Human xs AI
2. AI xs AI 

# Human xs AI 
    In this mode a human is collaborating with a Soft actor critic agent.
    The base command is:

    ```
        python td3_maze3d_train.py --config game\config\config_sac_No_TL.yaml --participant <name of participant>
    ```
    Where 
    1. --config -> The file where the setting of the Humman xs Ai experiment [the argument requires a path to config_sac_No_TL.yaml file]
    2. --participant -> name which the files would be saved. [The argumet requieres a String that it will use as a name to sertun folders and other files]

    
    Command for main Experiment is 
    ```
    python sac_maze3d_train.py --config game\config\config_sac_No_TL.yaml --participant Name --auto-alpha

    ```



# AI xs AI
    In this mode Two soft actor critic agents collaborate with each other.
    The base command is:
    ```
    python sac_maze3d_train.py --config game\config\config_sac_No_TL_Two_agents.yaml --participant  <name of participant>
    ```
    Where 
    1. --config -> The file where the setting of the Humman xs Ai experiment [the argument requires a path to config_sac_No_TL_Two_agents.yaml file]
    2. --participant -> name which the files would be saved. [The argumet requieres a String that it will use as a name to sertun folders and other files]

    Command for main Experiment is 
    ```
    python sac_maze3d_train.py --config game\config\config_sac_No_TL_two_agents.yaml --participant Name --auto-alpha

    ```

Some  usefull arguments are:

    * --auto-alpha -> Enables the alpha temperature to be trained during the gradiend updates [True if used false otherwise]
    * --alpha -> sets the value to alpha temperture if --auto-alpha is false [Float value, default is 0.05]
    * --load-buffer -> Loads a buffer from path using the argument --buffer-path-1, and completes a pretrained with said buffer. [True if used false otherwise]
    * --buffer-path-1 -> Path to a buffer in order to be used loaded when --load-buffer is True

Setting about the agent and the experiment can be change from the config files.

All paths must change from using \ to / if run on ubuntu.
