# Official Python image from the Docker Hub
FROM python:3.9-slim

# Setting up the working directory
WORKDIR /app


COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
