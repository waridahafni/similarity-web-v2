from flask import session, redirect
from functools import wraps


def role_required(allowed_roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'role' not in session or session['role'] not in allowed_roles:
                return redirect('/dashboard' if session.get('role') == 'user' else '/home')
            return func(*args, **kwargs)
        return wrapper
    return decorator
