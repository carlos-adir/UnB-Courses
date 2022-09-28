# First experiment

This folder contains the **experimental data**, the **algorithm** to treat the experimental data, the **output** of the algorithm and the **report** using these values.
The report is in portuguese but the file [plot_original_data.ipynb](https://github.com/carlos-adir/UnB-Courses/blob/main/vibracoes/first_experiment/plot_original_data.ipynb) shows the graphs of the experimental data, so you can use the same data as you want.

### The experimental data

The subfolders are

```
└── first_experiment
    ├── massa1
    │   └── test1-1
    ├── massa2
    │   ├── test1-2
    │   ├── test2-2
    │   └── test3-2
    ├── massa3
    │   ├── test1-3
    │   └── test2-3
    └── massa4
        ├── test1-4
        ├── test2-4
        └── test3-4
```

Inside each ```testX-Y``` folder, there are 3 files ```tps.txt```, ```frq.txt``` and ```mfc.txt```. There are better explained in the file [plot_original_data.ipynb](https://github.com/carlos-adir/UnB-Courses/blob/main/vibracoes/first_experiment/plot_original_data.ipynb).

### The algorithm

There is only one algorithm: the file [treat_experimental_data.py](https://github.com/carlos-adir/UnB-Courses/blob/main/vibracoes/first_experiment/treat_experimental_data.py).
The theory behind this file is in the python notebook [estimate_parameters.ipynb](https://github.com/carlos-adir/UnB-Courses/blob/main/vibracoes/estimate_parameters.ipynb).
So I strongly recommend you to see this notebook before reading the algorithm.

### The results

The output given by the ```treat_experimental_data.py``` is named by ```output_treatexpdata.txt```.
There you can find the estimated parameters (please see the notebook) for each test.

And then there is the report ```Vibracoes_Laboratorio1.pdf``` which is in portuguese and contains all the informations of ```output_tratexpdata.txt``` in the tables and the figures.