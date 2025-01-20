"""
Initial trial of FastAPI with Python

After running in terminal: fastapi dev omakase_fastapi.py
Test using URL: http://127.0.0.1:8000/omakase/?topic=Personal%20Development&subtopic=Self-Help&level=Beginner
"""

from fastapi import FastAPI
from pydantic import BaseModel
import omakase

app = FastAPI()

@app.get("/omakase/")
def gen_omakase(topic, subtopic, level):
    output = omakase.generate_omakase(topic, subtopic, level, n_options=3, verbose=False)
    return output

