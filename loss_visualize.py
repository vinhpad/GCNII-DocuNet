from matplotlib import pyplot as plt
losses = []
with open('losses.txt', 'r') as file:
    for loss in file:
        losses.append(float(loss))
plt.plot(losses)
print(losses)
plt.show()