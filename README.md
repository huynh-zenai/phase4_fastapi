# phase4_fastapi

## Build image and run
```bash
# CPU
docker build -f Dockerfile.cpu -t sd1-5-api:cpu .
docker run -p 8000:8000 sd1-5-api:cpu

# GPU
docker build -f Dockerfile.gpu -t sd1-5-api:gpu .
docker run --gpus all -p 8000:8000 sd1-5-api:gpu
```

## Testing
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"prompt":"a photo of an astronaut riding a horse on mars"}' \
     http://localhost:8000/txt2img | jq -r .image_base64 | base64 -d > out.png
```