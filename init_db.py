from myApplication import app, db
import logging as lg
from myApplication.models import User

with app.app_context():
    db.drop_all()
    db.create_all()
    user1 = User(username="ismaila", email="sanguesow@gmail.com")
    user1.set_password("motdepasse")
    db.session.add(user1)
    db.session.commit()
    lg.warning('Database has been created')
