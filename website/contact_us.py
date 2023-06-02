from flask import Blueprint, render_template

contact_us = Blueprint('contact_us', __name__)

@contact_us.route('/contact')
def contact():
    return render_template('contact.html')

@contact_us.route('/about')
def about():
    return render_template('about.html')
