from matplotlib import pyplot as plt

y = []

with open('s1.txt', 'r') as f:
    lines = f.readlines()

for line in lines[4:70]:
    fields = line.split()
    y.append(float(fields[2][:-1]))

plt.plot(range(len(y)), y, label='tutorial setting')

y = []

with open('s2.txt', 'r') as f:
    lines = f.readlines()

for line in lines[4:70]:
    fields = line.split()
    y.append(float(fields[2][:-1]))

plt.plot(range(len(y)), y, label='pefll setting')
plt.legend()
plt.title('validation dataset accuracy')
plt.show()
