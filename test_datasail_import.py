"""Explore DataSAIL's API and functionality."""
import datasail
from datasail import sail, routine, settings
import inspect

print("="*60)
print("DataSAIL Module Exploration")
print("="*60)

print("\n1. Main datasail module:")
print("-"*60)
for item in dir(datasail):
    if not item.startswith('_'):
        obj = getattr(datasail, item)
        print(f"  {item:20} - {type(obj).__name__}")

print("\n2. datasail.sail module (likely main API):")
print("-"*60)
for item in dir(sail):
    if not item.startswith('_'):
        obj = getattr(sail, item)
        print(f"  {item:20} - {type(obj).__name__}")

print("\n3. datasail.routine module:")
print("-"*60)
for item in dir(routine):
    if not item.startswith('_'):
        obj = getattr(routine, item)
        print(f"  {item:20} - {type(obj).__name__}")

print("\n4. Looking for main functions:")
print("-"*60)
if hasattr(sail, 'sail'):
    print("\nFound sail.sail() function!")
    print("Signature:", inspect.signature(sail.sail))
    print("\nDocstring:")
    print(sail.sail.__doc__)

if hasattr(routine, 'run'):
    print("\nFound routine.run() function!")
    print("Signature:", inspect.signature(routine.run))

print("\n5. Settings:")
print("-"*60)
for item in dir(settings):
    if not item.startswith('_') and item.isupper():
        print(f"  {item} = {getattr(settings, item)}")

