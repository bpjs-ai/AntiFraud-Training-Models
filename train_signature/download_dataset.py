"""Dataset download utilities for the YOLOv11 training pipeline.

This module provides a helper function to download a dataset from Roboflow.
It raises an informative error if the ``ROBOFLOW_API_KEY`` environment
variable is missing and returns a ``Path`` object pointing to the dataset
root directory.  Having the download logic in a standalone file makes it
possible to reuse dataset downloading across different training scripts.
"""

import os
from pathlib import Path

try:  # pragma: no cover
    from roboflow import Roboflow
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Roboflow package is not installed.\n"
        "Install it with 'pip install roboflow' before running this script."
    ) from e


def download_dataset(workspace: str, project: str, version: int, export_format: str) -> Path:
    """Download a dataset from Roboflow.

    Parameters
    ----------
    workspace : str
        The name of the Roboflow workspace.
    project : str
        The slug of the project within the workspace.
    version : int
        The version number to download.
    export_format : str
        The export format for the dataset. For YOLOv11 training, use "yolov11".

    Returns
    -------
    pathlib.Path
        A ``Path`` pointing to the directory where the dataset has been downloaded.

    Raises
    ------
    ValueError
        If the ``ROBOFLOW_API_KEY`` environment variable is not set.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Roboflow API key. Set the ROBOFLOW_API_KEY environment variable before running."
        )

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)
    proj = ws.project(project)
    ver = proj.version(version)
    dataset = ver.download(export_format)
    return Path(dataset.location)