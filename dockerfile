FROM python:3.10-slim

WORKDIR /app

# Copy only needed project folders
COPY capstone_project /app/capstone_project/


RUN pip install  -r capstone_project/requirements.txt

EXPOSE 5000

# Run Flask app
CMD ["python", "capstone_project/flask_app/app.py"]
