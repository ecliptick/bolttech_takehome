locals {
  name_prefix = "${var.project}-${var.environment}"
  azs         = slice(data.aws_availability_zones.available.names, 0, 2)
  # Aligns with notebooks 01–06 + app.config local paths → sync targets in S3.
  s3_prefix = {
    raw_xlsx               = "sources/claim_use_case_dataset.xlsx"
    enriched_csv           = "derived/claim_use_case_dataset_enriched.csv"
    pattern_strata_csv     = "derived/pattern_strata_denial_borderline.csv"
    synthetic_incremental  = "derived/synthetic_incremental_last_run.csv"
    eval_historical_logs   = "monitoring/eval_historical_logs.csv"
    eval_drift_logs        = "monitoring/eval_drift_logs.csv"
    model_joblib           = "models/approval_model.joblib"
    model_meta_json        = "models/approval_model_meta.json"
    prompts_glob_note      = "prompts/*.txt (mirror repo prompts/)"
  }
  # ECS tasks: private subnets only when NAT is enabled (docs: Fargate behind ALB in private subnets).
  ecs_subnet_ids           = var.enable_nat_gateway ? [for s in aws_subnet.private : s.id] : [for s in aws_subnet.public : s.id]
  ecs_assign_public_ip     = var.enable_nat_gateway ? false : true
  active_model_s3_uri      = "s3://${aws_s3_bucket.artifacts.bucket}/models/approval_model.joblib"
  active_model_meta_uri    = "s3://${aws_s3_bucket.artifacts.bucket}/models/approval_model_meta.json"
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

resource "aws_cloudwatch_log_group" "api" {
  name              = "/${local.name_prefix}/api"
  retention_in_days = 30

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}
