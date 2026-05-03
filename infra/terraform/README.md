# Terraform — AWS stack (`claim-approval-agent`)

This directory matches **`DESIGN.md` §1.2**, **`docs/AWS_DEPLOYMENT.md`**, and the production diagram in **`docs/PRODUCTION_ARCHITECTURE.md`**: VPC (public + private subnets), NAT (optional), **ALB**, **ECS Fargate**, **ECR**, versioned **S3** buckets, **CloudWatch Logs**, **SSM Parameter Store** model URIs, optional **Secrets Manager** for Gemini, and IAM wired for task + execution roles.

**Not in Terraform yet (still documentation-only)**:

- CI **metric-threshold gate** → promote artifacts (GitHub Actions / separate pipeline).
- **EventBridge → Step Functions** retrain state machine (and any Glue/Lambda trainers).
- **Route 53**, **ACM certificate** (HTTPS on ALB optional via `acm_certificate_arn` when you supply a cert in-region).
- **AWS WAF** (rate-limit LLM routes) — attach to ALB manually or extend this stack.
- A **hosted Prometheus / Grafana / AMP** fleet — the app exposes **`GET /metrics`** only; you still deploy a scraper or agent.

Terraform alone is enough for **static AWS shape + IAM**. You still typically add:

| Layer | Responsibility |
|--------|----------------|
| **This repo Terraform** | Network, ECS service, ALB, buckets, logs, parameters, IAM. |
| **CI/CD** | Build/push **ECR** image, `terraform plan/apply`, optional `aws s3 sync` for `models/` and datasets. |
| **Secrets hygiene** | Set real **Gemini** secret value (`inject_gemini_api_key_secret = true`) or inject via CI; never commit keys. |
| **Runtime scraper** | Prometheus-compatible scraper targeting `http(s)://<alb>/metrics` when enabled. |

## Layout

| File | Purpose |
|------|---------|
| `versions.tf` | Terraform + AWS provider pinning. |
| `variables.tf` | Region, VPC, ECS sizing, ACM, Gemini secret toggle, NAT toggle. |
| `data.tf` | Caller identity, AZ lookup. |
| `main.tf` | Locals, ECR, S3 buckets, CloudWatch log group. |
| `network.tf` | VPC, subnets, IGW, optional NAT / private routes. |
| `security_groups.tf` | ALB ↔ task rules. |
| `alb.tf` | ALB, target group, HTTP/HTTPS listeners. |
| `ssm.tf` | `/PROJECT/ENV/active-model-{,meta}-uri`. |
| `secrets.tf` | Optional Secrets Manager stub for Gemini. |
| `ecs.tf` | Cluster, IAM, task definition, Fargate service. |
| `outputs.tf` | ECR URL, ALB DNS, buckets, ECS names, parameter keys. |

## Before `terraform apply`

1. **Backend** — edit `versions.tf`: uncomment/configure `backend "s3" { … }` for remote state + locking (S3 + DynamoDB).
2. **Credentials** — `AWS_PROFILE`, environment variables, or OIDC-based role for CI (not defined here yet).
3. **Bootstrap artefacts in S3** — task definition injects `ACTIVE_MODEL_S3_URI` and `ACTIVE_MODEL_META_S3_URI` from Parameter Store pointing at **`s3://<artifacts_bucket>/models/approval_model.joblib`** (and `…_meta.json`). **Upload objects before relying on ECS**, or temporarily set `desired_task_count = 0` via `variables.tf` / CLI override (`-var`). The Docker image **bakes** `./artifacts`; at runtime **`app/ml/predict.py` overwrites local files when those env vars reference `s3://…`** (`boto3` + task IAM).
4. **ECR image** — `docker build -t repo_url:tag …` → `docker push`; default tag is **`latest`** (`ecr_image_tag` variable).

## Useful variables

| Variable | Default | Notes |
|----------|---------|-------|
| `enable_nat_gateway` | `true` | Set `false` to save ~USD 30/mo NAT cost; ECS tasks land in **public** subnets with `assign_public_ip` (dev-only posture). |
| `acm_certificate_arn` | `""` | If set, listener **443** terminates TLS and **HTTP 80** redirects to HTTPS. ACM cert must live in **this** region. |
| `inject_gemini_api_key_secret` | `false` | If `true`, creates Secrets Manager stub and wires `GEMINI_API_KEY`; replace placeholder value outside Terraform (console/CLI). |
| `desired_task_count` | `1` | Scale out for HA. |

## Commands

```bash
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform plan
terraform -chdir=infra/terraform apply
```

## Health check

Target group probes **`GET /health`** on port **8000** (container). Public URL uses ALB (**`alb_dns_name`** output) — default **HTTP 80 →** tasks.
