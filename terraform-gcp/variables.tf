variable "project_id" {
  description = "my GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "asia-southeast2"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "asia-southeast2-a"
}

variable "hf_token" {
  description = "Hugging Face Token (not needed for CPU/LightGBM plan)"
  type        = string
  sensitive   = true
  default     = "dummy"
}

variable "model_id" {
  description = "Model identifier (unused in CPU/LightGBM plan)"
  type        = string
  default     = "lgbm-creditcard-fraud"
}

variable "machine_type" {
  description = "GCE Machine Type — n2-standard-8 for CPU-only LightGBM plan"
  type        = string
  default     = "n2-standard-8"
}

