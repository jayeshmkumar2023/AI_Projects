#!/bin/bash
if [ "$(docker ps -aq -f name=qdrant)" ]; then
    if [ "$(docker ps -q -f name=qdrant)" ]; then
        echo "Qdrant container is already running."
    else
        echo "Starting existing Qdrant container..."
        docker start qdrant
    fi
else
    echo "Creating and starting new Qdrant container..."
    mkdir -p "$(pwd)/data/qdrant_storage"
    docker run -d \
      --name qdrant \
      -p 6333:6333 \
      -p 6334:6334 \
      -v "$(pwd)/data/qdrant_storage:/qdrant/storage" \
      qdrant/qdrant:latest
fi
