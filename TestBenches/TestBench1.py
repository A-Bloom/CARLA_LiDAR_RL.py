import traceback

for a in range(5):
    try:
        num1 = input("type here,")
        print(4/int(num1))
    except Exception:
        traceback.print_exc()


