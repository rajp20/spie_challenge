Running the code:

```python
python3.6 main.py {machine} {model} {optimizer}
```

Options for machine are 'cade' and 'local'. Cade option lets you run the code with more GPU memory.

Options for model are 'resnet', 'simple', and 'improved'.

Options for optimizer are 'adam' and 'sgd'.

Example command to run ResNet with SGD on CADE machine:
```python
python3.6 main.py cade resnet sgd
```