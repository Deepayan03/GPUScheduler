# 🧠 GPU Scheduler  
*A simple tool that watches your GPU and runs tasks only when the GPU is free.*

---

## 🚀 What This Project Will Be  
GPU Scheduler is being built to act like a **traffic controller for your GPU**.

The idea is simple:

- Check if the GPU is busy  
- If it is free → run the task  
- If it is busy → wait  
- Decide which task should run first  

This keeps your GPU from getting overloaded and helps heavy tasks run smoothly.

Think of it as a smart helper that says:  
**“Hold on… GPU is full… okay now you can run.”**

---

## Features built so far 
Right now, the project is still small.  
It only has the **GPU-watching part**:

- 🔍 Reads GPU usage on Apple Silicon using `powermetrics`  
- 🎮 Reads GPU usage on NVIDIA GPUs using `nvidia-smi`  
- 📸 Gives clean “snapshot” numbers  
- 🧩 Provides the base for the upcoming scheduler  


---

## 🌱 Future Plans (NOT Included Yet)  
These features are not built yet — but these are the main goals of GPU Scheduler.

### 🧵 1. Real Task Scheduler  
This will be the main brain of the project.  
It will:

- Look at GPU usage  
- Decide when to start or delay tasks  
- Prevent tasks from running at the same time  
- Keep GPU usage stable and smooth  

It’s like a teacher calling students one-by-one instead of all at once.

---

### ⏳ 2. Task Queue  
A **task queue** is like a waiting line for tasks.

- You add tasks to the line  
- Tasks stay in order  
- When the GPU is free, the next task runs  
- When the GPU is busy, tasks wait  

This stops tasks from overloading the GPU.

---

### ⏱️ 3. Real-Time Scheduling  
Real-time scheduling means the scheduler constantly checks GPU load.

- If GPU usage spikes → delay new tasks  
- If GPU usage drops → instantly run waiting tasks  
- Decisions happen **live**, not only once  

---

### ⭐ 4. Priority Control  
Some tasks are more important than others.

Priority control lets the scheduler choose:

- which task gets to run first  
- which tasks wait longer  
- which tasks jump ahead in the queue  


That’s priority scheduling.

---

## 📁 Files in the Project (So Far)

### **`monitor.py`**
- Main controller  
- Selects Apple/NVIDIA backend  
- Gives GPU usage snapshots  

### **`powermetrics_backend.py`**
- Gets GPU usage on Apple Silicon  
- Parses output from `powermetrics`  

### **`nvidia_backend.py`**
- Gets GPU usage on NVIDIA GPUs  
- Uses `nvidia-smi`  

### **`snapshot.py`**
- Defines a single GPU reading model  

### **`testMonitor.py`**
- Simple test script  
- Prints GPU snapshots to check if monitoring works  

---

## 🧩 Summary 
Right now → **It only watches GPU usage.**  
Later → **It will run tasks at the perfect time, like a smart GPU manager.**

---
