ARG BASE_IMAGE_LOCATION
ARG PROJECT_ID
ARG REPOSITORY_ID
# Use the base
FROM ${BASE_IMAGE_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_ID}/base-image:latest

# Set the working directory.
WORKDIR /app

# Copy local code to the container image.
COPY . .

RUN mkdir /app/creds && pwd && ls -latr

# Expose the port the function is listening on.
EXPOSE 8080

# Set the entrypoint for the function.
CMD ["functions-framework", "--target=worker", "--source=/app/main.py"]
