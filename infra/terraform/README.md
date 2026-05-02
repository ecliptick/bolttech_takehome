# Terraform stub (expand per account/region)

The files here are intentionally minimal placeholders. Before `terraform apply`, configure a remote **S3+DynamoDB** backend for state, set `region`, and wire your VPC.

Suggested first resources:

- `aws_ecr_repository` for API images  
- `aws_s3_bucket` with versioning for datasets and serialized models  
- `aws_cloudwatch_log_group` for ECS task logs  

Full ECS+Fargate+ALB modules are omitted to keep the prototype focused; use [terraform-aws-modules/terraform-aws-alb](https://github.com/terraform-aws-modules/terraform-aws-alb) and [terraform-aws-modules/terraform-aws-ecs](https://github.com/terraform-aws-modules/terraform-aws-ecs) when you are ready.
