from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import json
import os
import io
import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)

# Load model & scaler once
model = tf.keras.models.load_model(os.path.join(BASE, 'parkinsons_model.keras'))
scaler = joblib.load(os.path.join(BASE, 'scaler.pkl'))
df_full = pd.read_csv(os.path.join(BASE, 'parkinsons.csv'))

with open(os.path.join(BASE, 'precomputed_stats.json')) as f:
    STATS = json.load(f)

FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# In-memory prediction history
prediction_history = []


@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(STATS)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [float(data.get(f, 0)) for f in FEATURE_NAMES]
        arr = np.array(features).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        prob = float(model.predict(arr_scaled, verbose=0)[0][0])
        label = 'Parkinson\'s Detected' if prob > 0.5 else 'Healthy'
        confidence = prob if prob > 0.5 else 1 - prob

        entry = {
            'id': len(prediction_history) + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'result': label,
            'probability': round(prob * 100, 2),
            'confidence': round(confidence * 100, 2),
            'inputs': data
        }
        prediction_history.append(entry)

        return jsonify({
            'result': label,
            'probability': round(prob * 100, 2),
            'confidence': round(confidence * 100, 2),
            'is_parkinsons': prob > 0.5,
            'id': entry['id']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(prediction_history)


@app.route('/api/history/csv', methods=['GET'])
def download_history_csv():
    if not prediction_history:
        return jsonify({'error': 'No history'}), 404
    rows = []
    for h in prediction_history:
        row = {'id': h['id'], 'timestamp': h['timestamp'],
               'result': h['result'], 'probability': h['probability'],
               'confidence': h['confidence']}
        row.update(h['inputs'])
        rows.append(row)
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='prediction_history.csv'
    )


@app.route('/api/report', methods=['POST'])
def generate_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    data = request.json
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                             leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    BLUE = colors.HexColor('#1e40af')
    RED = colors.HexColor('#dc2626')
    GREEN = colors.HexColor('#16a34a')
    GRAY = colors.HexColor('#6b7280')
    LIGHT = colors.HexColor('#f0f4ff')

    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=22,
                                  textColor=BLUE, spaceAfter=4, alignment=TA_CENTER)
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=10,
                                textColor=GRAY, alignment=TA_CENTER, spaceAfter=12)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13,
                               textColor=BLUE, spaceBefore=16, spaceAfter=6)
    normal = ParagraphStyle('N', parent=styles['Normal'], fontSize=10, spaceAfter=4)
    bold = ParagraphStyle('B', parent=styles['Normal'], fontSize=10,
                           spaceAfter=4, fontName='Helvetica-Bold')

    story = []
    story.append(Paragraph("Parkinson's Disease Prediction Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}", sub_style))
    story.append(HRFlowable(width='100%', thickness=2, color=BLUE))
    story.append(Spacer(1, 0.4*cm))

    result = data.get('result', 'N/A')
    prob = data.get('probability', 0)
    conf = data.get('confidence', 0)
    result_color = RED if 'Parkinson' in result else GREEN

    # Result summary table
    result_data = [
        ['Diagnosis', 'Confidence', 'Probability'],
        [result, f"{conf:.1f}%", f"{prob:.1f}%"]
    ]
    t = Table(result_data, colWidths=[6*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BACKGROUND', (0,1), (-1,1), LIGHT),
        ('FONTNAME', (0,1), (0,1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,1), (0,1), result_color),
        ('FONTSIZE', (0,1), (-1,1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,1), [LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Model Details", h2_style))
    model_data = [
        ['Model', 'Deep Neural Network (DNN)'],
        ['Architecture', '128→64→32→16→1 (sigmoid)'],
        ['Test Accuracy', '94.87%'],
        ['ROC AUC Score', '98.62%'],
        ['Framework', 'TensorFlow / Keras'],
    ]
    mt = Table(model_data, colWidths=[5*cm, 9*cm])
    mt.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,0), (0,-1), LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.4*cm))

    # Input features
    inputs = data.get('inputs', {})
    if inputs:
        story.append(Paragraph("Patient Input Features", h2_style))
        feat_rows = [['Feature', 'Value']] + [[k, str(v)] for k, v in inputs.items()]
        ft = Table(feat_rows, colWidths=[8*cm, 6*cm])
        ft.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(ft)

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=1, color=GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<i>This report is generated by an AI model and is for informational purposes only. "
        "Consult a qualified medical professional for diagnosis and treatment.</i>",
        ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8,
                       textColor=GRAY, alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf', as_attachment=True,
                     download_name='parkinsons_report.pdf')


@app.route('/api/dataset/stats', methods=['GET'])
def dataset_stats():
    df = df_full.drop('name', axis=1)
    feats = FEATURE_NAMES
    stats = {}
    for f in feats:
        stats[f] = {
            'mean': round(float(df[f].mean()), 4),
            'std': round(float(df[f].std()), 4),
            'min': round(float(df[f].min()), 4),
            'max': round(float(df[f].max()), 4),
        }
    return jsonify({
        'feature_stats': stats,
        'total': len(df),
        'parkinsons': int(df['status'].sum()),
        'healthy': int((df['status'] == 0).sum())
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
