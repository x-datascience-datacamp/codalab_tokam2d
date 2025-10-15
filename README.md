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
