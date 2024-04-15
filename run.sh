python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

echo "TASK #1"
python -m task_1.test

echo "TASK #2"
python -m task_2.test

echo "TASK #3"
python -m task_3.test