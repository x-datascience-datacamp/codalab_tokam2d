# Tokam2D - Structure detection in fusion plasma simulations in codabench

TDB


To test the ingestion program, run:

```bash
python ingestion_program/ingestion.py --data-dir dev_phase/input_data/ --output-dir outputs  --submission-dir solution/
```


To test the scoring program, run:

```bash
python scoring_program/scoring.py --reference-dir dev_phase/reference_data/ --output-dir score_outputs  --prediction-dir outputs/
```


### Testing the docker image

To test the docker image locally, run:

```bash
docker run --rm -it -u root \
    -v "./ingestion_program":"/app/ingestion_program" \
    -v "./dev_phase/input_data":/app/input_data \
    -v "./ingestion_res":/app/output \
    -v "./solution":/app/ingested_program \
    --name ingestion tommoral/tokam2d:v1 \
    python /app/ingestion_program/ingestion.py

docker run --rm -it -u root \
    -v "./scoring_program":"/app/scoring_program" \
    -v "./dev_phase/reference_data":/app/input/ref \
    -v "./ingestion_res":/app/input/res \
    -v "./scoring_res":/app/output \
    --name scoring tommoral/tokam2d:v1 \
    python /app/scoring_program/scoring.py
```