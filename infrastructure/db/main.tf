terraform {
  required_providers {
    postgresql = {
      source  = "cyrilgdn/postgresql"
      version = "~> 1.19.0"
    }
  }
}

provider "postgresql" {
  host            = var.db_host
  port            = var.db_port
  database        = var.db_name
  username        = var.db_username
  password        = var.db_password
  sslmode         = "require" # Change to "disable" if you don't need SSL
  connect_timeout = 15
  superuser       = false     # Set to true if your user has superuser privileges
}

# Ensure that the "public" schema exists
resource "null_resource" "create_public_schema" {
  provisioner "local-exec" {
    command = "psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command \"CREATE TABLE IF NOT EXISTS public.page_raw (id CHAR(32) PRIMARY KEY, url TEXT NOT NULL, last_updated TIMESTAMP NOT NULL);\""

    environment = {
      PGPASSWORD = var.db_password
    }
  }
}

# Create the table "page_raw" if it doesn't already exist
resource "null_resource" "create_page_raw_table" {
  depends_on = [null_resource.create_public_schema]
  provisioner "local-exec" {
    command = "psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command \"CREATE TABLE IF NOT EXISTS public.page_raw (id CHAR(32), url TEXT NOT NULL, last_updated TIMESTAMP NOT NULL);\""
    environment = {
      PGPASSWORD = var.db_password
    }
  }
}

# Create the table "page_content" if it doesn't already exist
resource "null_resource" "create_page_content_table" {
  depends_on = [null_resource.create_public_schema]
  provisioner "local-exec" {
    command = "psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command \"CREATE TABLE IF NOT EXISTS public.page_content (id CHAR(32) PRIMARY KEY, url TEXT NOT NULL, html_content TEXT, extracted_title TEXT, extracted_content TEXT, fetched_at TIMESTAMP NOT NULL);\""
    environment = {
      PGPASSWORD = var.db_password
    }
  }
}


