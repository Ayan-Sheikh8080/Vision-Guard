# Run order (Git Bash in VS Code). Replace RollNo and <user>.
Take a screenshot (showing prompt, time, project name) after every numbered step.

## 0. Setup
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
python -c "import tensorflow as tf;print(tf.__version__,tf.config.list_physical_devices('GPU'))"
pip freeze > requirements-freeze.txt
cp .env.example .env        # fill in DagsHub values; then: git check-ignore .env  (proof)

## 1. Data (Task 1)
Download Kaggle dataset, unzip so you have data/raw/AiArtData and data/raw/RealArt
git init && git branch -M main && git remote add origin https://github.com/<user>/RollNo_A2_Ops.git
python -m src.audit && python -m src.eda
bash scripts/dvc_toy.sh      # run it from an EMPTY folder OUTSIDE this repo (copy the file there)
dvc init
dvc remote add -d origin https://dagshub.com/<user>/RollNo_A2_Ops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <token>
dvc add data/raw && git add data/raw.dvc data/.gitignore .dvc && git commit -m "track data"
dvc push
python -m src.split && git add splits && git commit -m "splits" && git push -u origin main

## 2. Tuner toy + tuning (Task 2.7)
python scripts/tuner_toy.py
python -m src.tune

## 3. MLflow toy (Task 3.1/3.2)
python scripts/mlflow_toy.py ; mlflow ui          # http://127.0.0.1:5000 -> compare 2 runs
python scripts/mlflow_toy.py --dagshub

## 4. Six+ runs (Tasks 2.3-2.6, 3.3-3.5)
python -m src.train --model cnn --run-name cnn_baseline
python -m src.train --model mobilenet --run-name mnv2_ft
python -m src.train --model mobilenet --autolog --run-name mnv2_autolog
python -m src.train --model mobilenet --no-aug --run-name mnv2_noaug
python -m src.train --model mobilenet --img 160 --run-name mnv2_160
python -m src.train --model efficientnet --run-name effb0
python -m src.errors mnv2_ft

## 5. Registry (Task 3.6)
python scripts/register.py <RUN_ID_A>
python scripts/register.py <RUN_ID_B> 2        # 2 = version number to alias champion
python scripts/load_champion.py some_image.jpg   # also writes models/champion.keras

## 6. Flask locally (Tasks 4,5)
set -a; source .env; set +a ; MODEL_PATH=models/champion.keras python -m app.app   # http://localhost:5000

## 7. Docker (Task 6)
docker compose up -d --build ; docker compose ps ; docker compose logs -f web
docker compose exec web whoami ; docker images ; docker history <image>

## 8. Cloud API (Task 7) - see chat instructions; then set API_URL / API_KEY in .env and `docker compose up -d`
