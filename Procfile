release: python manage.py collectstatic --noinput && python manage.py migrate
web: gunicorn Skripsi.wsgi:application --bind 0.0.0.0:$PORT
