#!/usr/bin/env bash

# Start FastAPI in the background on port 8001
uvicorn main:app --host 0.0.0.0 --port 8001 &

# Start Streamlit in the foreground on port 8000
python -m streamlit run app/streamlit_app.py --server.port 8000 --server.address 0.0.0.0