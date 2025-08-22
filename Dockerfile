# Use the base
#FROM ${BASE_IMAGE_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_ID}/base-image:latest
FROM python:3.11-slim
# Set the working directory.
WORKDIR /app

# Copy local code to the container image.
COPY . .
RUN pip install -r requirements.txt

RUN mkdir /app/creds && pwd && ls -latr

# Expose the port the function is listening on.
EXPOSE 8080

# Set the entrypoint for the function.
CMD ["functions-framework", "--target=worker", "--source=/app/main.py"]
