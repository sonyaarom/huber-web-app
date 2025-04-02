terraform {
  required_providers {
    postgresql = {
      source  = "cyrilgdn/postgresql"
      version = "~> 1.19.0"
    }
  }
}

provider "postgresql" {
  host            = "aws-0-eu-central-1.pooler.supabase.com"  
  port            = 5432         
  database        = var.db_name
  username        = var.db_username
  password        = var.db_password
  sslmode         = "disable"     
  connect_timeout = 15
  superuser       = false
}



resource "null_resource" "create_extension" {
  provisioner "local-exec" {
    command = "psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command \"CREATE EXTENSION IF NOT EXISTS vector;\""

    environment = {
      PGPASSWORD = var.db_password
    }
  }
}




resource "null_resource" "create_public_schema" {
  provisioner "local-exec" {
    command = "psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command \"CREATE SCHEMA IF NOT EXISTS public;\""

    environment = {
      PGPASSWORD = var.db_password
    }
  }
}


resource "null_resource" "create_page_raw_table" {
  depends_on = [null_resource.create_public_schema]
  provisioner "local-exec" {
    command = <<EOT
    psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command "
    CREATE TABLE IF NOT EXISTS public.page_raw (
        id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        last_updated TIMESTAMP NOT NULL,
        last_scraped TIMESTAMP NOT NULL,
        is_active BOOLEAN DEFAULT TRUE
    );"
    EOT

    environment = {
      PGPASSWORD = var.db_password
    }
  }
}


resource "null_resource" "create_page_content_table" {
  depends_on = [null_resource.create_public_schema]
  provisioner "local-exec" {
    command = <<EOT
    psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command "
    CREATE TABLE IF NOT EXISTS public.page_content (
        id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        html_content TEXT,
        extracted_title TEXT,
        extracted_content TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        last_updated TIMESTAMP NOT NULL,
        last_scraped TIMESTAMP NOT NULL
    );"
    EOT

    environment = {
      PGPASSWORD = var.db_password
    }
  }
}

resource "null_resource" "create_page_keywords_table" {
  depends_on = [null_resource.create_public_schema]
  provisioner "local-exec" {
    command = <<EOT
    psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command "
    CREATE TABLE IF NOT EXISTS public.page_keywords (
        id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        last_modified TIMESTAMP NOT NULL,
        tokenized_text TSVECTOR,
        raw_text TEXT,
        last_scraped TIMESTAMP NOT NULL
    );"
    EOT

    environment = {
      PGPASSWORD = var.db_password
    }
  }
}


resource "null_resource" "create_page_embeddings_table" {
  depends_on = [null_resource.create_extension, null_resource.create_public_schema]
  provisioner "local-exec" {
    command = <<EOT
    psql --host=${var.db_host} --port=${var.db_port} --username=${var.db_username} --dbname=${var.db_name} --command "
    CREATE TABLE IF NOT EXISTS public.page_embeddings (
        id TEXT NOT NULL,
        split_id TEXT NOT NULL,
        url TEXT,
        chunk_text TEXT,
        embedding_vector VECTOR(1536),
        PRIMARY KEY (id, split_id),
        last_scraped TIMESTAMP NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_page_embeddings_vector 
    ON public.page_embeddings 
    USING hnsw (embedding_vector vector_l2_ops);
    "
    EOT

    environment = {
      PGPASSWORD = var.db_password
    }
  }
}




resource "postgresql_schema" "public" {
  name  = "public"
}
resource "local_file" "bm25_search_function" {
  content  = <<-EOT
    CREATE OR REPLACE FUNCTION bm25_search(
      p_query TEXT,
      p_limit INTEGER DEFAULT 10,
      p_k1 NUMERIC DEFAULT 1.2,
      p_b NUMERIC DEFAULT 0.75,
      p_table TEXT DEFAULT 'page_keywords'
    )
    RETURNS TABLE(
      id TEXT,
      score NUMERIC,
      raw_text TEXT,
      url TEXT
    )
    AS $$
    DECLARE
      sql_query TEXT;
    BEGIN
      sql_query := format($f$
        WITH params AS (
          SELECT
            count(*) AS total_docs,
            COALESCE((SELECT avg(length(tokenized_text::TEXT)) FROM %I), 1) AS avg_dl
        ),
        raw_scores AS (
          SELECT
            pk.id::TEXT AS id,
            pk.url::TEXT AS url,
            pk.raw_text::TEXT AS raw_text,
            SUM(
              CASE
                WHEN ts_rank_cd(pk.tokenized_text, plainto_tsquery('simple', $1)) > 0
                THEN ts_rank_cd(pk.tokenized_text, plainto_tsquery('simple', $1), 32) -- BM25 Ranking
                ELSE 0
              END
            ) AS bm25_score
          FROM %I pk
          CROSS JOIN params p
          WHERE pk.tokenized_text @@ plainto_tsquery('simple', $1)  -- Proper multi-word search
          GROUP BY pk.id, pk.url, pk.raw_text
          HAVING SUM(
            CASE
              WHEN ts_rank_cd(pk.tokenized_text, plainto_tsquery('simple', $1)) > 0
              THEN ts_rank_cd(pk.tokenized_text, plainto_tsquery('simple', $1), 32)
              ELSE 0
            END
          ) > 0
        )
        SELECT
          c.id,
          ROUND(c.bm25_score::NUMERIC, 3) AS score,
          c.raw_text,
          c.url
        FROM raw_scores c
        ORDER BY c.bm25_score DESC
        LIMIT $2;
      $f$, p_table, p_table);

      RETURN QUERY EXECUTE sql_query
        USING p_query, p_limit;
    END;
    $$ LANGUAGE plpgsql;
  EOT
  filename = "${path.module}/sql/bm25_search.sql"
}

resource "postgresql_function" "bm25_search" {
  name     = "bm25_search"
  database = var.db_name  
  schema   = postgresql_schema.public.name
  body     = local_file.bm25_search_function.content
  
  parameter {
    name = "p_query"
    type = "text"
  }
  
  parameter {
    name = "p_limit"
    type = "integer"
    default = 10
  }
  
  parameter {
    name = "p_k1"
    type = "numeric"
    default = 1.2
  }
  
  parameter {
    name = "p_b"
    type = "numeric"
    default = 0.75
  }
  
  parameter {
    name = "p_table"
    type = "text"
    default = "page_keywords"
  }
  
  returns = "TABLE(id text, score numeric, raw_text text, url text)"
  language = "plpgsql"
  security_definer = false
  
  depends_on = [
    local_file.bm25_search_function
  ]
}