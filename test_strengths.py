import os

for strength in [i * 0.025 for i in range(1, 25)]:
    print('-' * 50)
    print(f'{strength=}')

    os.system(f'python encode.py --strength {strength}')
    os.system('python decode.py')

    print(f'{strength=}')
    print('-' * 50)
    print()
