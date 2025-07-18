# Conf.py
bind = "0.0.0.0:8001"
workers = 3
worker_class = "uvicorn.workers.UvicornWorker"
keepalive = 120
timeout = 120
errorlog = "logs/error.log" 
accesslog = "access.log"
loglevel = "info"