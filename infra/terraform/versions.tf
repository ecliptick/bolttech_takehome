terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # backend "s3" {
  #   bucket         = "your-tf-state-bucket"
  #   key            = "claim-approval-agent/terraform.tfstate"
  #   dynamodb_table = "terraform-state-lock"
  #   encrypt        = true
  #   region         = "eu-north-1"
  # }
}

provider "aws" {
  region = var.aws_region
}
