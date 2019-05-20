Prerequisite:
    - python 3.7
    - Numpy 1.16.2
    - Scipy 1.2.1
    - matplotlib 3.0.2
    - GUI frameworks: pygtk, gobject, tkinter, PySide, PyQt4, wx (all optional, but one is required for an interactive GUI)

running the code instructions:
    The folder tree must look like this:
        - main.py
        - NNClassifier.py
        - functions.py
        - data (folder) (read permission required)
            - train.csv
            - validate.csv
            - test.csv
        - figures (folder) (write permission required)
    The folder that contains the main.py file should enable write permissions
    in order for the program to generate the output.txt file.

    One should execute the code by running the following command line from Unix-like terminal:
        >> python main.py
