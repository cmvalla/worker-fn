# Use the base
FROM europe-west1-docker.pkg.dev/spanner-demo-bengal/my-docker-repo/base-image:latest

# Set the working directory.
WORKDIR /app

# Copy local code to the container image.
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the function is listening on.
EXPOSE 8080

# Set the entrypoint for the function.
CMD ["functions-framework", "--target=worker", "--source=main.py"]
