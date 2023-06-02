from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from . import db, views


from werkzeug.security import generate_password_hash, check_password_hash


auth = Blueprint('auth', __name__)


@auth.route('/login', methods = ['GET','POST'])
def login():
    data = request.form
    print(data)
    return render_template('login.html',user='Thash')

@auth.route('/logout')
def logout():
    return "<p>Logout</p>"

@auth.route('/sign-up', methods = ['GET','POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        fname = request.form.get('first_name')
        sname = request.form.get('second_name')
        password1 = request.form.get('password1')
        password2 = request.form.get("password2")
        print(type(fname))
        if len(email) < 4:
            flash('Email Invalid', category='error')
        elif len(fname) < 2:
            flash('First Name must be larger than 2 characters',category='error')
        elif password1 != password2:
            flash('Passwords do not match',category='error')
        else:
            new_user = User(email=email, first_name=fname, second_name=sname, password = generate_password_hash(password1,'sha256'))
            db.session.add(new_user)
            db.session.commit()

            flash('Sign Up Complete!',category='success')
            return redirect(url_for(views.home))
    return render_template('signup.html')