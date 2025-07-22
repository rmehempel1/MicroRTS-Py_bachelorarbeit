import pandas as pd
import matplotlib.pyplot as plt
import ast

name="uas2"
df = pd.read_csv(f"./csv/{name}.csv", on_bad_lines='skip')
window = 20  # Gleitfenstergröße
print(df.columns)

def clean_reward(value):
    if isinstance(value, str) and "[" in value:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return float(parsed[0])
        except:
            return float('nan')  # bei Fehler NaN
    try:
        return float(value)
    except:
        return float('nan')

df = df.drop_duplicates(subset='episode', keep='first')
df["reward"] = df["reward"].apply(clean_reward)
df["SmoothedReward"] = df["reward"].rolling(window=window).mean()
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward", color="black")
ax1.scatter(df["episode"], df["reward"], color="blue", alpha=0.3, s=5)
ax1.plot(df["episode"], df["SmoothedReward"], label=f"Reward (Moving Avg {window})", color="black", linewidth=2)
ax1.tick_params(axis='y', labelcolor="black")

# Zweite Achse für Epsilon (rechte y-Achse)
ax2 = ax1.twinx()
ax2.set_ylabel("Epsilon", color="red")
ax2.plot(df["episode"], df["epsilon"], color="red", label="Epsilon", linestyle="--")
ax2.tick_params(axis='y', labelcolor="red")

# Titel, Legenden und Grid
plt.title(f"Entwicklung des Rewards und Epsilon über Episoden Model:{name}")
fig.tight_layout()
plt.grid(True)
plt.show()


"""LOSS"""
# Smoothed Loss berechnen

df["SmoothedLoss"] = df["loss"].rolling(window=window).mean()

# === Plot 1: Loss + Epsilon über Episoden ===
fig2, ax3 = plt.subplots(figsize=(12, 4))
ax3.set_xlabel("Episode")
ax3.set_ylabel("Loss", color="green")
ax3.plot(df["episode"], df["SmoothedLoss"], color="green", label=f"Loss (Moving Avg {window})", linewidth=2)
ax3.tick_params(axis='y', labelcolor="green")

ax4 = ax3.twinx()
ax4.set_ylabel("Epsilon", color="red")
ax4.plot(df["episode"], df["epsilon"], color="red", linestyle="--", label="Epsilon")
ax4.tick_params(axis='y', labelcolor="red")

plt.title(f"Entwicklung von Loss und Epsilon über Episoden – Model: {name}")
fig2.tight_layout()
plt.grid(True)
plt.show()

# === Plot 3: Reward + Loss + Epsilon ===
fig, ax1 = plt.subplots(figsize=(12, 6))

# === Reward ===
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward", color="black")
ax1.plot(df["episode"], df["SmoothedReward"], label="Reward (Smoothed)", color="black", linewidth=2)
ax1.tick_params(axis='y', labelcolor="black")

# === Loss (rechte y-Achse) ===
ax2 = ax1.twinx()
ax2.set_ylabel("Loss", color="green")
ax2.plot(df["episode"], df["SmoothedLoss"], label="Loss (Smoothed)", color="green", linewidth=2)
ax2.tick_params(axis='y', labelcolor="green")

# === Epsilon auf dritter Achse (zweite rechte Achse, oben, invertiert) ===
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))  # Abstand zur rechten Kante
ax3.set_ylabel("Epsilon", color="red")
ax3.plot(df["episode"], df["epsilon"], label="Epsilon", color="red", linestyle=":", linewidth=1.5, alpha=0.7)
ax3.tick_params(axis='y', labelcolor="red")  # Damit Epsilon von oben nach unten verläuft
ax3.grid(False)     # Kein zusätzliches Grid

# === Legenden kombinieren ===
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc="upper right")

ax1.grid(True)
ax2.grid(False)
ax3.grid(False)

plt.title(f"Kombinierter Verlauf von Reward, Loss und Epsilon – getrennte Skalen – Model: {name}")
fig.tight_layout()
plt.show()