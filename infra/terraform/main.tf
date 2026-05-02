# Minimal AWS skeleton — fill provider + backend before apply.
terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # backend "s3" { ... }
}

variable "environment" {
  type        = string
  description = "e.g. dev, staging, prod"
  default     = "dev"
}

variable "project" {
  type        = string
  description = "Name prefix for resources"
  default     = "claim-approval-agent"
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type        = string
  description = "AWS region"
  default     = "eu-north-1"
}

locals {
  name_prefix = "${var.project}-${var.environment}"
  # Aligns with notebooks 01–06 + app.config local paths → sync targets in S3.
  s3_prefix = {
    raw_xlsx              = "sources/claim_use_case_dataset.xlsx"
    enriched_csv          = "derived/claim_use_case_dataset_enriched.csv"
    pattern_strata_csv      = "derived/pattern_strata_denial_borderline.csv"
    synthetic_incremental   = "derived/synthetic_incremental_last_run.csv"
    eval_historical_logs    = "monitoring/eval_historical_logs.csv"
    eval_drift_logs         = "monitoring/eval_drift_logs.csv"
    model_joblib            = "models/approval_model.joblib"
    model_meta_json         = "models/approval_model_meta.json"
    prompts_glob_note       = "prompts/*.txt (mirror repo prompts/)"
  }
}

# --- Container registry (push images built from repo Dockerfile)
resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# --- Model / dataset versioning (datasets, joblib snapshots, prompts sync)
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name_prefix}-ml-artifacts-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Datasets / CSV extracts / monitoring sample logs (notebook `data/` directory)
resource "aws_s3_bucket" "datasets" {
  bucket = "${local.name_prefix}-datasets-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project     = var.project
    Environment = var.environment
    Purpose     = "notebook-data-and-derived-csv"
  }
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Container logs (FastAPI in `app/main.py`; ship stdout here from ECS/Fargate or similar)
resource "aws_cloudwatch_log_group" "api" {
  name              = "/${local.name_prefix}/api"
  retention_in_days = 30

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}

data "aws_caller_identity" "current" {}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "artifacts_bucket" {
  value       = aws_s3_bucket.artifacts.bucket
  description = "Store approval_model.joblib, approval_model_meta.json; mirror prompts/ if desired."
}

output "datasets_bucket" {
  value       = aws_s3_bucket.datasets.bucket
  description = "Store enriched CSV, strata tables, synthetic_incremental_last_run, eval_* logs."
}

output "recommended_s3_object_prefixes" {
  value       = local.s3_prefix
  description = "Suggested keys when syncing local notebook/repo outputs to S3."
}

output "cloudwatch_log_group_api" {
  value       = aws_cloudwatch_log_group.api.name
  description = "Destination for API container stdout (prediction/explain structured logs)."
}
