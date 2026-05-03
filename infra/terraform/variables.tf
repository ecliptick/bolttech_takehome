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

variable "aws_region" {
  type        = string
  description = "AWS region"
  default     = "eu-north-1"
}

variable "vpc_cidr" {
  type        = string
  description = "VPC IPv4 CIDR"
  default     = "10.0.0.0/16"
}

variable "enable_nat_gateway" {
  type        = bool
  description = "Required for tasks in private subnets to reach Gemini/ECR HTTPS. Disabling puts ECS tasks in public subnets (cheap dev-only; weaker isolation)."
  default     = true
}

variable "acm_certificate_arn" {
  type        = string
  description = "If non-empty, add HTTPS listener on 443 + redirect HTTP 80→443."
  default     = ""
}

variable "ecr_image_tag" {
  type        = string
  description = "Tag pushed for the API image (bootstrap: push latest after terraform apply)."
  default     = "latest"
}

variable "desired_task_count" {
  type        = number
  description = "Fargate service desired count."
  default     = 1
}

variable "fargate_cpu" {
  type        = number
  description = "Fargate CPU units (e.g. 512 = 0.25 vCPU; see ECS task definition)."
  default     = 512
}

variable "fargate_memory" {
  type        = number
  description = "Fargate memory (MiB)."
  default     = 1024
}

variable "inject_gemini_api_key_secret" {
  type        = bool
  description = "If true, GEMINI_API_KEY is loaded from Secrets Manager secret created here (placeholder string on first apply; update out-of-band)."
  default     = false
}
