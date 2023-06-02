from flask import Blueprint, render_template

western_prediction = Blueprint('western_prediction', __name__)

@western_prediction.route('/western')
def western():
    return render_template('western.html')