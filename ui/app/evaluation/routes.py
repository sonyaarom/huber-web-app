import os
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Optional, List
from flask import Blueprint, request, render_template, flash, redirect, url_for, current_app, send_file, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from functools import wraps

from hubert.db.postgres_storage import PostgresStorage
from hubert.config import settings

# Configure logging first
logger = logging.getLogger(__name__)

# Optional imports - these modules require additional dependencies like wandb
try:
    from hubert.retriever_evaluate.evaluate_retrievers import RetrieverEvaluator
    RETRIEVER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Retriever evaluation not available: {e}")
    RETRIEVER_AVAILABLE = False

try:
    from hubert.prompt_evaluation.evaluators.wandb_evaluation import run_evaluation as run_prompt_evaluation
    PROMPT_EVALUATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Prompt evaluation not available: {e}")
    PROMPT_EVALUATION_AVAILABLE = False

try:
    from hubert.prompt_evaluation.prompts.prompt_templates import PromptFactory
    PROMPT_FACTORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Prompt factory not available: {e}")
    PROMPT_FACTORY_AVAILABLE = False

# Create Blueprint
bp = Blueprint('evaluation', __name__, url_prefix='/evaluation')

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
DEFAULT_CSV_FILES = {
    'qa_pairs.csv': 'Default QA pairs for evaluation',
    'qa_pairs_filtered.csv': 'Filtered QA pairs for evaluation'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_default_csv_files():
    """Get available default CSV files from the project"""
    csv_files = []
    possible_paths = [
        'assets/csv/',
        'assets/',
        './',
        'hubert/assets/csv/',
        'hubert/assets/'
    ]
    
    for base_path in possible_paths:
        for filename in DEFAULT_CSV_FILES.keys():
            full_path = os.path.join(base_path, filename)
            if os.path.exists(full_path):
                csv_files.append({
                    'filename': filename,
                    'path': full_path,
                    'description': DEFAULT_CSV_FILES[filename]
                })
    return csv_files

@bp.route('/')
@login_required
@admin_required
def dashboard():
    """Main evaluation dashboard"""
    try:
        storage = PostgresStorage()
        embedding_tables = storage.get_embedding_tables()
        
        # Get available prompts (your existing Langfuse-based prompt types)
        available_prompts = ['base', 'medium', 'advanced'] if PROMPT_FACTORY_AVAILABLE else []
        
        # Get available Langfuse prompt IDs from settings
        langfuse_prompts = getattr(settings, 'langfuse_prompt_ids', {}) if PROMPT_FACTORY_AVAILABLE else {}
        
        # Get available models
        available_models = ['llama', 'openai'] if PROMPT_EVALUATION_AVAILABLE else []
        
        # Get default CSV files
        default_csvs = get_default_csv_files()
        
        return render_template('evaluation/dashboard.html', 
                             title='Evaluation Dashboard',
                             embedding_tables=embedding_tables,
                             available_prompts=available_prompts,
                             available_models=available_models,
                             default_csvs=default_csvs,
                             langfuse_prompts=langfuse_prompts,
                             retriever_available=RETRIEVER_AVAILABLE,
                             prompt_evaluation_available=PROMPT_EVALUATION_AVAILABLE,
                             prompt_factory_available=PROMPT_FACTORY_AVAILABLE)
    except Exception as e:
        logger.error(f"Error loading evaluation dashboard: {e}")
        flash(f"Error loading evaluation page: {e}", 'danger')
        return render_template('evaluation/dashboard.html', 
                             title='Evaluation Dashboard',
                             embedding_tables=[],
                             available_prompts=[],
                             available_models=[],
                             default_csvs=[],
                             langfuse_prompts={},
                             retriever_available=RETRIEVER_AVAILABLE,
                             prompt_evaluation_available=PROMPT_EVALUATION_AVAILABLE,
                             prompt_factory_available=PROMPT_FACTORY_AVAILABLE)

@bp.route('/retriever', methods=['POST'])
@login_required
@admin_required
def run_retriever_evaluation():
    """Run retriever evaluation"""
    if not RETRIEVER_AVAILABLE:
        flash('Retriever evaluation is not available. Missing dependencies (wandb, etc.).', 'danger')
        return redirect(url_for('evaluation.dashboard'))
    
    try:
        # Get form data
        selected_tables = request.form.getlist('embedding_tables')
        csv_source = request.form.get('csv_source')
        use_reranker = request.form.get('use_reranker') == 'on'
        use_hybrid_search = request.form.get('use_hybrid_search') == 'on'
        hybrid_alpha = float(request.form.get('hybrid_alpha', 0.5))
        top_k = int(request.form.get('top_k', 10))
        
        if not selected_tables:
            flash('Please select at least one embedding table to evaluate.', 'warning')
            return redirect(url_for('evaluation.dashboard'))
        
        # Determine CSV file path
        csv_path = None
        if csv_source == 'upload':
            if 'csv_file' not in request.files:
                flash('No file uploaded.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
            
            file = request.files['csv_file']
            if file.filename == '':
                flash('No file selected.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                csv_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(csv_path)
            else:
                flash('Invalid file type. Please upload a CSV file.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
        else:
            # Use default CSV
            default_csv = request.form.get('default_csv')
            if default_csv:
                csv_path = default_csv
            else:
                # Try to find qa_pairs_filtered.csv or qa_pairs.csv
                for csv_file in get_default_csv_files():
                    if 'filtered' in csv_file['filename']:
                        csv_path = csv_file['path']
                        break
                if not csv_path:
                    csv_path = 'qa_pairs_filtered.csv'  # Default fallback
        
        if not csv_path or not os.path.exists(csv_path):
            flash('CSV file not found. Please upload a file or ensure default files exist.', 'warning')
            return redirect(url_for('evaluation.dashboard'))
        
        logger.info(f"Starting retriever evaluation with tables: {selected_tables}")
        logger.info(f"Using CSV file: {csv_path}")
        logger.info(f"Parameters: reranker={use_reranker}, hybrid={use_hybrid_search}, alpha={hybrid_alpha}, top_k={top_k}")
        
        # Create evaluator
        evaluator = RetrieverEvaluator(
            qa_pairs_path=csv_path,
            top_k=top_k,
            wandb_project=f"retriever-evaluation-{datetime.now().strftime('%Y%m%d')}",
            specific_tables=selected_tables,
            use_reranker=use_reranker,
            use_hybrid_search=use_hybrid_search,
            hybrid_alpha=hybrid_alpha
        )
        
        # Run evaluation
        evaluator.run_evaluation()
        evaluator.cleanup()
        
        flash('Retriever evaluation completed successfully! Check your Weights & Biases dashboard for results.', 'success')
        return redirect(url_for('evaluation.dashboard'))
        
    except Exception as e:
        logger.error(f"Error during retriever evaluation: {e}")
        flash(f"An error occurred during retriever evaluation: {str(e)}", 'danger')
        return redirect(url_for('evaluation.dashboard'))

@bp.route('/prompt', methods=['POST'])
@login_required
@admin_required
def run_prompt_evaluation():
    """Run prompt evaluation"""
    if not PROMPT_EVALUATION_AVAILABLE:
        flash('Prompt evaluation is not available. Missing dependencies (wandb, etc.).', 'danger')
        return redirect(url_for('evaluation.dashboard'))
    
    try:
        # Get form data
        selected_prompts = request.form.getlist('prompts')
        selected_models = request.form.getlist('models')
        num_questions = request.form.get('num_questions')
        csv_source = request.form.get('csv_source')
        
        if not selected_prompts:
            flash('Please select at least one prompt type.', 'warning')
            return redirect(url_for('evaluation.dashboard'))
        
        if not selected_models:
            flash('Please select at least one model.', 'warning')
            return redirect(url_for('evaluation.dashboard'))
        
        # Convert num_questions to int or None
        try:
            num_questions = int(num_questions) if num_questions and num_questions.strip() else None
        except ValueError:
            num_questions = None
        
        # Determine CSV file path
        csv_path = None
        if csv_source == 'upload':
            if 'csv_file' not in request.files:
                flash('No file uploaded.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
            
            file = request.files['csv_file']
            if file.filename == '':
                flash('No file selected.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                csv_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(csv_path)
            else:
                flash('Invalid file type. Please upload a CSV file.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
        else:
            # Use default CSV
            default_csv = request.form.get('default_csv')
            if default_csv:
                csv_path = default_csv
            else:
                # Try to find qa_pairs_filtered.csv or qa_pairs.csv
                for csv_file in get_default_csv_files():
                    if 'filtered' in csv_file['filename']:
                        csv_path = csv_file['path']
                        break
                if not csv_path:
                    csv_path = 'qa_pairs_filtered.csv'  # Default fallback
        
        if not csv_path or not os.path.exists(csv_path):
            flash('CSV file not found. Please upload a file or ensure default files exist.', 'warning')
            return redirect(url_for('evaluation.dashboard'))
        
        logger.info(f"Starting prompt evaluation with prompts: {selected_prompts}")
        logger.info(f"Using models: {selected_models}")
        logger.info(f"Using CSV file: {csv_path}")
        logger.info(f"Number of questions: {num_questions}")
        
        # Create output directory
        output_dir = f'assets/csv/evaluation_results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Run evaluation
        output_file = run_prompt_evaluation(
            csv_file=csv_path,
            num_questions=num_questions,
            prompts=selected_prompts,
            generation_types=selected_models,
            wandb_project=f"prompt-evaluation-{datetime.now().strftime('%Y%m%d')}",
            wandb_entity=settings.wandb_entity,
            output_dir=output_dir
        )
        
        flash(f'Prompt evaluation completed successfully! Results saved to {output_file}. Check your Weights & Biases dashboard for detailed results.', 'success')
        return redirect(url_for('evaluation.dashboard'))
        
    except Exception as e:
        logger.error(f"Error during prompt evaluation: {e}")
        flash(f"An error occurred during prompt evaluation: {str(e)}", 'danger')
        return redirect(url_for('evaluation.dashboard'))

@bp.route('/combined', methods=['POST'])
@login_required
@admin_required
def run_combined_evaluation():
    """Run both retriever and prompt evaluation"""
    if not RETRIEVER_AVAILABLE or not PROMPT_EVALUATION_AVAILABLE:
        flash('Combined evaluation requires both retriever and prompt evaluation modules. Missing dependencies (wandb, etc.).', 'danger')
        return redirect(url_for('evaluation.dashboard'))
    
    try:
        # Get form data
        selected_tables = request.form.getlist('embedding_tables')
        selected_prompts = request.form.getlist('prompts')
        selected_models = request.form.getlist('models')
        csv_source = request.form.get('csv_source')
        use_reranker = request.form.get('use_reranker') == 'on'
        use_hybrid_search = request.form.get('use_hybrid_search') == 'on'
        hybrid_alpha = float(request.form.get('hybrid_alpha', 0.5))
        top_k = int(request.form.get('top_k', 10))
        num_questions = request.form.get('num_questions')
        
        if not selected_tables or not selected_prompts or not selected_models:
            flash('Please select tables, prompts, and models for combined evaluation.', 'warning')
            return redirect(url_for('evaluation.dashboard'))
        
        # Convert num_questions to int or None
        try:
            num_questions = int(num_questions) if num_questions and num_questions.strip() else None
        except ValueError:
            num_questions = None
        
        # Determine CSV file path (same logic as above)
        csv_path = None
        if csv_source == 'upload':
            if 'csv_file' not in request.files:
                flash('No file uploaded.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
            
            file = request.files['csv_file']
            if file.filename == '':
                flash('No file selected.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                csv_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(csv_path)
            else:
                flash('Invalid file type. Please upload a CSV file.', 'warning')
                return redirect(url_for('evaluation.dashboard'))
        else:
            # Use default CSV
            default_csv = request.form.get('default_csv')
            if default_csv:
                csv_path = default_csv
            else:
                # Try to find qa_pairs_filtered.csv or qa_pairs.csv
                for csv_file in get_default_csv_files():
                    if 'filtered' in csv_file['filename']:
                        csv_path = csv_file['path']
                        break
                if not csv_path:
                    csv_path = 'qa_pairs_filtered.csv'  # Default fallback
        
        if not csv_path or not os.path.exists(csv_path):
            flash('CSV file not found. Please upload a file or ensure default files exist.', 'warning')
            return redirect(url_for('evaluation.dashboard'))
        
        logger.info("Starting combined evaluation")
        
        # Run retriever evaluation first
        logger.info("Starting retriever evaluation phase")
        retriever_evaluator = RetrieverEvaluator(
            qa_pairs_path=csv_path,
            top_k=top_k,
            wandb_project=f"combined-evaluation-retriever-{datetime.now().strftime('%Y%m%d')}",
            specific_tables=selected_tables,
            use_reranker=use_reranker,
            use_hybrid_search=use_hybrid_search,
            hybrid_alpha=hybrid_alpha
        )
        retriever_evaluator.run_evaluation()
        retriever_evaluator.cleanup()
        
        # Run prompt evaluation
        logger.info("Starting prompt evaluation phase")
        output_dir = f'assets/csv/evaluation_results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        output_file = run_prompt_evaluation(
            csv_file=csv_path,
            num_questions=num_questions,
            prompts=selected_prompts,
            generation_types=selected_models,
            wandb_project=f"combined-evaluation-prompt-{datetime.now().strftime('%Y%m%d')}",
            wandb_entity=settings.wandb_entity,
            output_dir=output_dir
        )
        
        flash(f'Combined evaluation completed successfully! Retriever results in Weights & Biases, prompt results saved to {output_file}.', 'success')
        return redirect(url_for('evaluation.dashboard'))
        
    except Exception as e:
        logger.error(f"Error during combined evaluation: {e}")
        flash(f"An error occurred during combined evaluation: {str(e)}", 'danger')
        return redirect(url_for('evaluation.dashboard'))

@bp.route('/upload_csv', methods=['POST'])
@login_required
@admin_required
def upload_csv():
    """Handle CSV file upload"""
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Validate CSV structure
            try:
                df = pd.read_csv(filepath)
                required_columns = ['question', 'answer', 'id']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    os.remove(filepath)  # Clean up invalid file
                    return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}'}), 400
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'filepath': filepath,
                    'rows': len(df)
                })
            except Exception as e:
                os.remove(filepath)  # Clean up invalid file
                return jsonify({'error': f'Invalid CSV format: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
    
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@bp.route('/preview_csv')
@login_required
@admin_required
def preview_csv():
    """Preview CSV file content"""
    try:
        csv_path = request.args.get('path')
        if not csv_path or not os.path.exists(csv_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(csv_path)
        
        # Return first 5 rows and basic info
        preview_data = {
            'columns': df.columns.tolist(),
            'rows_total': len(df),
            'preview_rows': df.head(5).to_dict('records')
        }
        
        return jsonify(preview_data)
    
    except Exception as e:
        logger.error(f"Error previewing CSV: {e}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@bp.route('/results')
@login_required
@admin_required
def view_results():
    """View evaluation results"""
    # This could be expanded to show historical results, logs, etc.
    return render_template('evaluation/results.html', title='Evaluation Results') 