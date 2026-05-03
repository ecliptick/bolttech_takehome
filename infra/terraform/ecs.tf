resource "aws_ecs_cluster" "main" {
  name = substr(replace("${local.name_prefix}-cluster", "_", "-"), 0, 255)

  setting {
    name  = "containerInsights"
    value = var.environment == "prod" ? "enabled" : "disabled"
  }

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name_prefix}-ecs-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name               = "${local.name_prefix}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

locals {
  ecs_secrets = concat(
    [
      {
        name      = "ACTIVE_MODEL_S3_URI"
        valueFrom = aws_ssm_parameter.active_model_uri.arn
      },
      {
        name      = "ACTIVE_MODEL_META_S3_URI"
        valueFrom = aws_ssm_parameter.active_model_meta_uri.arn
      },
    ],
    var.inject_gemini_api_key_secret ? [{
      name      = "GEMINI_API_KEY"
      valueFrom = aws_secretsmanager_secret.gemini[0].arn
    }] : [],
  )

  ecs_execution_policy_statements = concat(
    [
      {
        Sid      = "SsmRuntimeParams"
        Effect   = "Allow"
        Action   = ["ssm:GetParameters"]
        Resource = [aws_ssm_parameter.active_model_uri.arn, aws_ssm_parameter.active_model_meta_uri.arn]
      },
    ],
    var.inject_gemini_api_key_secret ? [{
      Sid      = "SecretsManagerGemini"
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.gemini[0].arn]
    }] : [],
  )
}

resource "aws_iam_role_policy" "ecs_execution_extras" {
  name = "${local.name_prefix}-ecs-exec-extras"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = local.ecs_execution_policy_statements
  })
}

resource "aws_iam_role_policy" "ecs_task_s3_models" {
  name = "${local.name_prefix}-task-s3"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListArtifactBucketModelsPrefix"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.artifacts.arn
        Condition = {
          StringLike = { "s3:prefix" = ["models/*"] }
        }
      },
      {
        Sid      = "GetArtifactModels"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.artifacts.arn}/models/*"
      },
    ]
  })
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.fargate_cpu
  memory                   = var.fargate_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = "api"
    image     = "${aws_ecr_repository.api.repository_url}:${var.ecr_image_tag}"
    essential = true
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.api.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "api"
      }
    }
    environment = [
      { name = "DISABLE_PROMETHEUS", value = "0" },
    ]
    secrets = local.ecs_secrets
  }])
}

resource "aws_ecs_service" "api" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_task_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.ecs_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = local.ecs_assign_public_ip
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_target_group.api]

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}
