from flask import Flask, request, render_template, send_file
from utils import process_document, generate_report, reconstruct_document

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = f'sample_files/{file.filename}'
        file.save(file_path)
        analysis_result, cosine_sim = process_document(file_path)
        report = generate_report(analysis_result, cosine_sim)
        new_doc_path = reconstruct_document(file_path, analysis_result)
        return render_template('index.html', report=report, new_doc_path=new_doc_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f'sample_files/{filename}', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
