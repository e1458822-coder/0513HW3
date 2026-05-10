# 📘 Deep Reinforcement Learning - Homework 3: DQN and its Variants

本儲存庫包含了深度強化學習課程 Homework 3 的完整實作與探討報告。主要目標為在 `Gridworld` 環境中，從最基礎的 Naive DQN 開始，逐步實作並比較不同的 DQN 變體與優化技巧，以應對不同複雜度的環境模式。

---

## 📂 專案結構

- **`Gridworld.py` / `GridBoard.py`**: 環境建構腳本，包含 `static`、`player` 與 `random` 模式的定義與遊戲邏輯。
- **`run_hw3_1.py`**: HW3-1 實作，包含 Naive DQN 與 Experience Replay DQN (針對 `static` 模式)。
- **`run_hw3_2.py`**: HW3-2 實作，包含 Double DQN 與 Dueling DQN (針對 `player` 模式)。
- **`run_hw3_3.py`**: HW3-3 實作，以 PyTorch Lightning 重構 DQN，並加入多種訓練優化技巧 (針對 `random` 模式)。
- **`HW3_1.md`**: HW3-1 短篇理解與探討報告。
- **`HW3_2.md`**: HW3-2 模型改良探討報告。
- **`HW3_3.md`**: HW3-3 訓練技巧與優化報告。

---

## 🧠 HW3-1: Naive DQN & Experience Replay Buffer
- **環境設定**: `static` 模式 (玩家、目標、障礙物位置皆固定)
- **探討重點**: 
  - 實作最基礎的 Naive DQN。
  - 探討 Naive DQN 因為「樣本高度相關」與「目標不穩定」導致的缺點。
  - 實作 Experience Replay Buffer，解釋其如何打破數據序列的時間相關性並提高樣本利用率，從而穩定神經網路收斂。

## ⚖️ HW3-2: Enhanced DQN Variants
- **環境設定**: `player` 模式 (目標、障礙物固定，玩家初始位置隨機)
- **探討重點**:
  - 由於初始狀態變得複雜，此部分實作並比較了兩種進階 DQN 架構：
    1. **Double DQN**: 將「動作選擇」與「價值評估」解耦，解決標準 DQN 中極易發生的 Q 值高估偏差 (Overestimation Bias) 問題。
    2. **Dueling DQN**: 將網路末端拆分成 Value 與 Advantage 兩條路徑，在多數動作無關緊要的狀態下，能更有效地學習狀態本身的價值。

## 🔁 HW3-3: PyTorch Lightning DQN & Training Tips
- **環境設定**: `random` 模式 (玩家、目標、障礙物位置全數隨機)
- **探討重點**:
  - 將訓練邏輯封裝至 **PyTorch Lightning** 框架中，提升程式碼可讀性與維護性。
  - 整合三大關鍵訓練技巧以穩定隨機環境下的崩潰問題：
    1. **Gradient Clipping (梯度裁剪)**: 防止因環境變化劇烈導致的梯度爆炸。
    2. **Learning Rate Scheduling (學習率排程)**: 隨著訓練步數增加，自動衰減學習率以幫助後期精細微調。
    3. **Huber Loss (Smooth L1 Loss)**: 替換掉原本的 MSE，提升模型對離群值的魯棒性。

---

## 🚀 執行方式

請確保在已安裝 PyTorch 的 Anaconda 環境（如 `DRL`）中執行以下指令。
若要執行 HW3-3，請確定環境中已安裝 `pytorch-lightning`。

```bash
# 執行 HW3-1
python run_hw3_1.py

# 執行 HW3-2
python run_hw3_2.py

# 執行 HW3-3
python run_hw3_3.py
```
