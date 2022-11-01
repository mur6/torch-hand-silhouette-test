# torch-hand-silhouette-test
## Install
```
$ python3 -m venv .venv
$ source .venv/bin/activate.fish
(.venv) $ pip install -r requirements/main.txt
(.venv) $ pip install -r requirements/dev.txt
```

## Run
```
PYTHONPATH=. python src/model.py
PYTHONPATH=. python src/loss.py
PYTHONPATH=. python src/utils/image.py
PYTHONPATH=. python src/utils/render.py
PYTHONPATH=. python src/utils/mano.py
```
