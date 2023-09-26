#!/usr/bin/env python

from prefect import flow, task

@task
def hello_world(name):
    print(f"hello world: {name}")

@task
def bye_world(name):
    print(f"bye world: {name}")

@flow(name="test flow")
def test_flow(names=["john", "smith"]):
    for name in names:
        hello_world(name)
        bye_world(name)

if __name__ == "__main__":
    test_flow()