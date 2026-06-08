from datasets import load_dataset
from rich import print

from ardevo.dataset.icarus import deserialize_task


def main():
  print("Ardevo")

  stream = load_dataset("Ardea/Icarus-dataset", name="rung_1", streaming=True, split="train")
  task = deserialize_task(next(iter(stream)))

  print(task)


if __name__ == "__main__":
  main()
