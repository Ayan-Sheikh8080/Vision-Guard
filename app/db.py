"""Task 5: request history in SQL (Postgres in Compose, SQLite locally)."""
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from app import config
engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)

def init_db(retries=10):
    for i in range(retries):
        try:
            with engine.begin() as c:
                c.execute(text("""CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY %s, created_at TIMESTAMP, filename TEXT, label TEXT,
                    confidence REAL, model_name TEXT, model_version TEXT, threshold REAL)"""
                    % ("AUTOINCREMENT" if engine.dialect.name == "sqlite" else "GENERATED ALWAYS AS IDENTITY")))
            return
        except Exception as e:
            print("DB not ready:", e, flush=True); time.sleep(2)
    raise RuntimeError("database unreachable")

def add(filename, label, conf, model_name, model_version, thr):
    with engine.begin() as c:
        c.execute(text("INSERT INTO predictions (created_at,filename,label,confidence,model_name,model_version,threshold)"
                       " VALUES (:t,:f,:l,:c,:m,:v,:th)"),
                  dict(t=datetime.utcnow(), f=filename, l=label, c=conf, m=model_name, v=model_version, th=thr))

def recent(n=50):
    with engine.connect() as c:
        return c.execute(text("SELECT created_at,filename,label,confidence,model_name,model_version "
                              "FROM predictions ORDER BY created_at DESC LIMIT :n"), dict(n=n)).fetchall()
