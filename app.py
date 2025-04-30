from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
import openai_cotizacion as oac

app = Flask(__name__, static_folder='static')
CORS(app)

def cleanup_uploads_folder():
    """Clean up the uploads folder by removing all its contents"""
    try:
        if os.path.exists('uploads'):
            # First try to remove all files in the directory
            for filename in os.listdir('uploads'):
                file_path = os.path.join('uploads', filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove file {file_path}: {str(e)}")
            
            # Then try to remove the directory itself
            try:
                os.rmdir('uploads')
            except Exception as e:
                print(f"Warning: Could not remove uploads directory: {str(e)}")
                
        # Create new uploads directory
        os.makedirs('uploads', exist_ok=True)
        print("Uploads folder initialized successfully")
        
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")
        # Ensure uploads directory exists even if cleanup fails
        os.makedirs('uploads', exist_ok=True)

file_paths = ["./uploads/propuesta_A.pdf"]
proposal_type = "tecnica"
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.form.get('message')
        files = request.files.getlist('file')
        quotations = []

        if files:
            # Create uploads directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)
            
            # Process each file
            for file in files:
                # Create a unique filename to avoid conflicts
                filename = os.path.join('uploads', file.filename)
                try:
                    # Save the file
                    file.save(filename)
                    
                    # Generate quotation for this file
                    quotation = oac.generate_cotizacion([filename], proposal_type)
                    quotations.append(f"Cotización para {file.filename}:\n{quotation}\n")
                    
                except Exception as e:
                    quotations.append(f"Error generando cotización para {file.filename}: {str(e)}\n")
            
            # Combine all quotations
            combined_quotations = "\n".join(quotations)
            
            response = {
                'response': f'Cotizaciones generadas:\n{combined_quotations}',
                'similar_templates': [],
                'projectName': 'Multiple Quotations'
            }
        else:
            response = {
                'response': 'No se recibieron archivos para cotizar.',
                'similar_templates': [],
                'projectName': 'No Files'
            }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview():
    try:
        markdown = request.json.get('markdown')
        return f"<div>{markdown}</div>"
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    # Clean up uploads folder on server startup
    cleanup_uploads_folder()
    app.run(debug=True, port=5000)