import os
import warnings
from .mlflow import MLflowLoggerCallback, _NoopModule
from types import ModuleType
from typing import Dict, Optional, Union
from ray.tune.experiment import Trial
from ray.air._internal.dagshub import _DagsHubLoggerUtil
from ray.util.annotations import PublicAPI

from ray.air import session

try:
    import dagshub
except:
    dagshub = None

class DagsHubLoggerCallback(MLflowLoggerCallback):

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        *,
        registry_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict] = None,
        tracking_token: Optional[str] = None,
        save_artifact: bool = False,
        dagshub_repository: Optional[str] = None,
        log_mlflow_only: bool = False,
    ):
        super(DagsHubLoggerCallback, self).__init__(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name,
            tags=tags,
            tracking_token=tracking_token,
            save_artifact=save_artifact
        )

        if dagshub is None:
            raise RuntimeError(
            "dagshub was not found - please install with `pip install dagshub`"
        )

        if self.tracking_uri and not self.tracking_uri.startswith("https://dagshub.com"):
            raise ConnectionError("Not the support uri hosted by DagsHub")

        dagshub_auth = os.getenv("DAGSHUB_USER_TOKEN")
        if dagshub_auth:
            dagshub.auth.add_app_token(dagshub_auth)

        repo_name, repo_owner = None, None
        if dagshub_repository:
            repo_name, repo_owner = self.splitter(dagshub_repository)
        else:
            if self.tracking_uri:
                repo_owner = self.tracking_uri.split(os.sep)[-2],
                repo_name = self.tracking_uri.split(os.sep)[-1].replace(".mlflow", ""),

        token = dagshub.auth.get_token()
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        if not repo_name or not repo_owner:
            repo_name, repo_owner = self.splitter(input("Please insert your repository owner_name/repo_name:"))

        if "MLFLOW_TRACKING_URI" not in os.environ or "dagshub" not in os.getenv("MLFLOW_TRACKING_URI"):
            dagshub.init(repo_name=repo_name, repo_owner=repo_owner)
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        self.mlflow_util = _DagsHubLoggerUtil(repo_owner=repo_owner, repo_name=repo_name, mlflow_only=log_mlflow_only)

    @staticmethod
    def splitter(repo):
        splitted = repo.split("/")
        if len(splitted) != 2:
            raise ValueError(f"Invalid input, should be owner_name/repo_name, but got {repo} instead")
        return splitted[1], splitted[0]
    
    def log_trial_end(self, trial: "Trial", failed: bool = False):
        run_id = self._trial_runs[trial]

        # Log the artifact if set_artifact is set to True.
        if self.should_save_artifact:
            self.mlflow_util.save_artifacts(run_id=run_id, dir=trial.local_path)
            if not self.mlflow_util.mlflow_only:
                self.mlflow_util.upload(run_id=run_id if run_id else "")
        # Stop the run once trial finishes.
        status = "FINISHED" if not failed else "FAILED"
        self.mlflow_util.end_run(run_id=run_id, status=status)