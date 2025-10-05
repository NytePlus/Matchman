![](doc/dataFlow.png)

```bash
PYTHONPATH=. python src/deploy/backend.py
```

```bash
PYTHONPATH=. gunicorn -w 1 -b 0.0.0.0:5000 --worker-class eventlet wsgi:app
```