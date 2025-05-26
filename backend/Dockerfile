# Use an official Python runtime as a parent image.
# python:3.13 is a good choice for a stable Python environment.
FROM python:3.13

# Set the working directory in the container to /app.
# All subsequent commands will be executed relative to this directory.
WORKDIR /app

# Copy the requirements.txt file into the container at /app.
# This step is done early to take advantage of Docker's layer caching.
# If only requirements.txt changes, only this layer and subsequent ones rebuild.
COPY requirements.txt .

# Install any Python packages specified in requirements.txt.
# --no-cache-dir prevents pip from saving download caches, reducing image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend project directory into the container at /app.
# The '.' means copy from the current context directory (where Dockerfile is).
COPY . /app

# Expose port 8000. This informs Docker that the container listens on the specified
# network ports at runtime. Django's default development server runs on 8000,
# and Gunicorn will also listen on this port.
EXPOSE 8000

# Run Gunicorn server. Gunicorn is a production-ready WSGI HTTP Server.
# "0.0.0.0:8000" makes the server accessible from outside the container.
# "real_estate_chatbot_project.wsgi:application" specifies the WSGI entry point
# for your Django project.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "real_estate_chatbot_project.wsgi:application"]