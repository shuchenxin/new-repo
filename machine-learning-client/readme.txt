Machine Learning client container
=================================

Build the worker image (from the repository root):

    docker build -t ml-client -f machine-learning-client/Dockerfile machine-learning-client

Run it against your MongoDB instance (replace the URI if needed):

    docker run --rm --env MONGO_URI="mongodb://mongodb:27017/ml_audio" ml-client

The container executes `python -m app.worker`, which continuously polls the
`tasks` collection, processes audio from GridFS, and writes results back to
MongoDB. The trained model and scaler artefacts in `data/fma_metadata/` are
copied into the image so no extra volume mounts are required unless you want to
override them. Use the `MONGO_URI` environment variable to point at an external
MongoDB deployment or another container on the same Docker network.
