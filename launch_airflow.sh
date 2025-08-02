#!/bin/bash

# Set dags folder path
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"

uv run airflow standalone
