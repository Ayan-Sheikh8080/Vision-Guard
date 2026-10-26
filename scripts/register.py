"""Task 3.6. Usage: python scripts/register.py <run_id> [alias_version]
Register twice (two different run ids) => 2 versions, then set alias champion."""
import sys, mlflow
from dotenv import load_dotenv
load_dotenv()
run_id = sys.argv[1]
v = mlflow.register_model(f"runs:/{run_id}/model", "VisionGuard")
print("registered version", v.version)
if len(sys.argv) > 2:
    mlflow.MlflowClient().set_registered_model_alias("VisionGuard", "champion", int(sys.argv[2]))
    print("alias champion ->", sys.argv[2])
