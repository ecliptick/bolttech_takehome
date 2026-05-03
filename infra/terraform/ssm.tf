# DESIGN.md §1.2 — SSM pointer for promoted model objects (URI only; pull-at-runtime belongs in application code).

resource "aws_ssm_parameter" "active_model_uri" {
  name  = "/${var.project}/${var.environment}/active-model-uri"
  type  = "String"
  value = local.active_model_s3_uri

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "active_model_meta_uri" {
  name  = "/${var.project}/${var.environment}/active-model-meta-uri"
  type  = "String"
  value = local.active_model_meta_uri

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}
