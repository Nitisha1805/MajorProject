from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from sqlalchemy.orm import sessionmaker

# base model class
Base = declarative_base()


    
# utility functions
def open_db():
    engine = create_engine('sqlite:///project.db', echo=True)
    session =  sessionmaker(bind=engine)
    return session()

def add_to_db(object):
    db = open_db()
    db.add(object)
    db.commit()
    db.close()

if __name__ == "__main__":
    # create engine
    # mysql settings
    # database_name = 'projectdb'
    # host = 'localhost'
    # username = 'root'
    # password = 'root'
    # port = 3306
    # engine = engine.create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}')
    # demo
    engine = create_engine('sqlite:///project.db', echo=True)
    Base.metadata.create_all(engine)