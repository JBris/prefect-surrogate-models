from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def hello_world(name):
    return f"hello world: {name}"


@task
def bye_world(name):
    return f"bye world: {name}"


@flow(
    name="My SubFlow",
    description="My flow using SequentialTaskRunner",
    task_runner=SequentialTaskRunner(),
)
def test_subflow(name: str):
    hello_world(name)
    bye_world(name)
