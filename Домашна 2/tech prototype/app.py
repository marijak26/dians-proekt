import os
import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

DATA_FOLDER = 'data'

@app.route('/company/all')
def list_files():
    try:
        files = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    except FileNotFoundError:
        files = []
    return render_template('companies.html', files=files)


@app.route('/company/<filename>')
def display_file(filename):
    try:
        file_path = os.path.join(DATA_FOLDER, filename+".csv")
        df = pd.read_csv(file_path)

        data_html = df.to_html(classes='table table-bordered', index=False)
        return render_template('file_contents.html', filename=filename, data_html=data_html)
    except FileNotFoundError:
        return f"File {filename} not found.", 404


if __name__ == '__main__':
    app.run(debug=True)
