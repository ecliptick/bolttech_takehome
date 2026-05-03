resource "aws_secretsmanager_secret" "gemini" {
  count = var.inject_gemini_api_key_secret ? 1 : 0
  name  = "${local.name_prefix}/gemini-api-key"

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "gemini_placeholder" {
  count         = var.inject_gemini_api_key_secret ? 1 : 0
  secret_id     = aws_secretsmanager_secret.gemini[0].id
  secret_string = "REPLACE_ME_AFTER_APPLY"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
