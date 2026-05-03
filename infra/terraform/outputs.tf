output "ecr_repository_url" {
  description = "Authenticate with ECR then docker push IMAGE:TAG."
  value       = aws_ecr_repository.api.repository_url
}

output "alb_dns_name" {
  description = "Public ALB hostname (DNS) — POST to http(s):///v1/predict via this host once healthy."
  value       = aws_lb.api.dns_name
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}

output "artifacts_bucket" {
  description = "Store approval_model.joblib + approval_model_meta.json under models/. Sync before scaling ECS."
  value       = aws_s3_bucket.artifacts.bucket
}

output "datasets_bucket" {
  description = "Enriched CSV, strata tables, synthetic_incremental_last_run, eval_* logs."
  value       = aws_s3_bucket.datasets.bucket
}

output "recommended_s3_object_prefixes" {
  value       = local.s3_prefix
  description = "Suggested keys when syncing local notebook/repo outputs to S3."
}

output "cloudwatch_log_group_api" {
  value       = aws_cloudwatch_log_group.api.name
  description = "API stdout (structured prediction/explain JSON lines)."
}

output "active_model_parameter_names" {
  description = "SSM Parameter Store keys used by ECS (values default to URIs under artifacts bucket)."
  value = {
    joblib_uri = aws_ssm_parameter.active_model_uri.name
    meta_uri   = aws_ssm_parameter.active_model_meta_uri.name
  }
}

output "gemini_secret_arn" {
  description = "Non-empty only when inject_gemini_api_key_secret=true — set SecretString to your real Gemini key after apply."
  value       = var.inject_gemini_api_key_secret ? aws_secretsmanager_secret.gemini[0].arn : null
}
