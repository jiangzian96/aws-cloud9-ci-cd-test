install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

	
lint:
	pylint --disable=R,C train.py &&\
		pylint --disable=R,C test.py
		

all:
	install lint
	
run:
	python train.py &&\
		python test.py
	