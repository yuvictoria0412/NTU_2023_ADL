import matplotlib.pyplot as plt

train_loss = [{
      "epoch": 0.04,
      "learning_rate": 0.0001904904904904905,
      "loss": 1.3317,
      "step": 100
    },
    {
      "epoch": 0.08,
      "learning_rate": 0.0001804804804804805,
      "loss": 1.2936,
      "step": 200
    },
    {
      "epoch": 0.12,
      "learning_rate": 0.00017047047047047048,
      "loss": 1.1847,
      "step": 300
    },
    {
      "epoch": 0.16,
      "learning_rate": 0.00016046046046046047,
      "loss": 1.2077,
      "step": 400
    },
    {
      "epoch": 0.2,
      "learning_rate": 0.00015045045045045046,
      "loss": 1.247,
      "step": 500
    },
    {
      "epoch": 0.24,
      "learning_rate": 0.00014044044044044044,
      "loss": 1.2019,
      "step": 600
    },
    {
      "epoch": 0.28,
      "learning_rate": 0.00013043043043043043,
      "loss": 1.1656,
      "step": 700
    },
    {
      "epoch": 0.32,
      "learning_rate": 0.00012042042042042043,
      "loss": 1.2031,
      "step": 800
    },
    {
      "epoch": 0.36,
      "learning_rate": 0.00011041041041041043,
      "loss": 1.1791,
      "step": 900
    },
    {
      "epoch": 0.4,
      "learning_rate": 0.00010040040040040039,
      "loss": 1.1468,
      "step": 1000
    },
    {
      "epoch": 0.44,
      "learning_rate": 9.039039039039039e-05,
      "loss": 1.1038,
      "step": 1100
    },
    {
      "epoch": 0.48,
      "learning_rate": 8.038038038038038e-05,
      "loss": 1.1571,
      "step": 1200
    },
    {
      "epoch": 0.52,
      "learning_rate": 7.047047047047048e-05,
      "loss": 1.1588,
      "step": 1300
    },
    {
      "epoch": 0.56,
      "learning_rate": 6.0460460460460465e-05,
      "loss": 1.0823,
      "step": 1400
    },
    {
      "epoch": 0.6,
      "learning_rate": 5.0450450450450445e-05,
      "loss": 1.0911,
      "step": 1500
    },
    {
      "epoch": 0.64,
      "learning_rate": 4.044044044044044e-05,
      "loss": 1.0185,
      "step": 1600
    },
    {
      "epoch": 0.68,
      "learning_rate": 3.0430430430430436e-05,
      "loss": 1.0615,
      "step": 1700
    },
    {
      "epoch": 0.72,
      "learning_rate": 2.0420420420420422e-05,
      "loss": 1.0885,
      "step": 1800
    },
    {
      "epoch": 0.76,
      "learning_rate": 1.0410410410410411e-05,
      "loss": 1.0509,
      "step": 1900
    },
    {
      "epoch": 0.8,
      "learning_rate": 4.004004004004004e-07,
      "loss": 1.0663,
      "step": 2000
    }]

loss = [d["loss"] for d in train_loss]
print(loss)

# x = list(range(len(loss)))

# # Create the plot
# plt.plot(x, loss)

# # Optionally set the title and labels
# plt.title('Loss over time')
# plt.xlabel('Step')
# plt.ylabel('Loss')

# # Display the plot
# plt.show()