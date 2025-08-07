from ml_orchestrator import MLOrchestrator
from timeit import timeit

orch = MLOrchestrator()
test_image = open("test_image.jpg", "rb").read()

print("Cold start:", timeit(lambda: orch.analyze_toy_image(test_image), number=1)
print("Warm performance:", timeit(lambda: orch.analyze_toy_image(test_image), number=10)
