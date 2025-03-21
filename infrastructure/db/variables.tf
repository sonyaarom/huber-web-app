variable "db_host" {
  type        = string
  description = "The hostname of the PostgreSQL server"
}

variable "db_port" {
  type        = number
  description = "The port number of the PostgreSQL server"
}

variable "db_name" {
  type        = string
  description = "The name of the PostgreSQL database"
}

variable "db_username" {
  type        = string
  description = "The PostgreSQL username"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "The PostgreSQL password"
}
