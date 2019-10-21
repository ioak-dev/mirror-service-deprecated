# Start Server locally

## First time
pip install virtualenv
pip install virtualenvwrapper-win
pip install -r requirements.txt
mkvirtualenv <some name for the environment> (eg: mkvirtualenv local)
python manage.py runserver

## Every time
workon <environment name> (eg: workon local)
python manage.py runserver
