# VIDO

Added support of Sacred library: https://github.com/IDSIA/sacred

Usage:  

From Python:   
- run main.py => without preprocessing (change code to enable preprocessing)  

From command line:
- python main.py => without preprocessing  
- python main.py with vido.full_preprocessing  => full preprocessing (lemmatize, lowercase, remove_stop_words, stem)  
- python main.py with vido.lemmatize=True (vido.lowercase/vido.remove_stop_words/vido.stem) => partial preprocessing  
- python main.py print_config => print configuration of variables and seed  
