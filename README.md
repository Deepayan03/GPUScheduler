# 🚀 GPU Scheduler  
## Load-Aware GPU Task Control for Single-Machine Systems

> GPUs are powerful. They are not polite.  
> GPU Scheduler makes them behave.

---

## 📖 The Story

You’re training a deep learning model.

Your GPU is already at 92%.

You quickly launch a small inference script.  
Or a visualization tool.  
Or another experiment.

Suddenly:

• Training slows down  
• Memory allocation fails  
• CUDA errors appear  
• System responsiveness drops  

The GPU didn’t fail because it’s weak.  
It failed because it was overloaded.

Unlike CPUs, GPUs are designed for throughput — not multitasking.  
They do not gracefully handle overload in single-machine environments.

And on most personal systems, nothing regulates *when* GPU tasks start.

That’s the real problem.

---

## 🎯 The Core Problem

When GPU utilization approaches full capacity:

• Memory allocation becomes fragile  
• Kernel launch latency increases  
• Throughput becomes unpredictable  
• Even lightweight tasks may fail  

Operating systems handle CPU scheduling well.  
But GPU execution start timing is largely unmanaged.

Cluster schedulers like Slurm solve this at scale.  
They are heavyweight and built for multi-node clusters.

There is a gap for lightweight scheduling on standalone machines.

---

## 💡 The Idea

GPU Scheduler asks one simple question before launching any task:

> “Is it safe to start this right now?”

If yes → allow execution.  
If no → wait.

There is no kernel interruption.  
No GPU driver modification.  
No forced preemption.

Just intelligent admission control.

---

## 🏗 System Architecture

<p align="center">
  <img src="assets/architecture.png" width="850"/>
</p>



Core components:

User / Application  
Priority Task Queue  
Scheduler Engine  
Monitoring Daemon  
Resource Estimator  
GPU Hardware  
Execution Logs  

The daemon continuously monitors GPU utilization.  
The scheduler makes admission decisions.  
The queue manages waiting tasks by priority.

The GPU itself remains untouched.

---

## 🔄 How It Works

1. A background daemon continuously tracks GPU utilization  
2. It detects both scheduled and externally launched GPU tasks  
3. The system estimates how much GPU a new task will require  
4. Effective load is calculated with safety headroom  
5. If total load stays below threshold (85–90%), the task runs  
6. Otherwise, the task waits  
7. When GPU load drops, waiting tasks are admitted  

This is preventive scheduling — not reactive fixing.

---

## ⚙ Scheduling Strategy

Priority-Based Scheduling  
Load-Aware Admission Control  
Non-Preemptive Execution  

Tasks are ordered by importance.

Priority influences queue order, but never overrides safety thresholds.

Once a task starts, it runs without interruption.

---

## 📈 Why Cap Utilization at 85–90%?

Running GPUs at absolute 100% sounds efficient.  
In practice, it can be fragile.

Sustained 100% utilization increases:

• Power draw  
• Temperature  
• Memory contention  
• Clock fluctuation under thermal or power constraints  

Maintaining utilization headroom:

• Reduces sustained thermal stress  
• Minimizes clock instability under long workloads  
• Absorbs short load spikes  
• Improves performance consistency  
• Prevents unpredictable slowdowns  

The goal is not limiting performance.  
The goal is maintaining stability under pressure.

---

## 🖥 CPU vs GPU Scheduling Reality

CPUs:
• Lightweight preemption  
• Small execution state  
• Frequent context switching  

GPUs:
• Thousands of parallel threads  
• Large execution state  
• Expensive preemption  
• Throughput-optimized design  

Stopping a CPU task is easy.  
Stopping a GPU task mid-kernel is complex and costly.

So instead of interrupting GPU work,  
GPU Scheduler controls when tasks begin.

---

## 🔬 Optional: Cooperative Training Support

For long-running ML workloads:

Training can run inside a wrapper that supports checkpointing.

When necessary:

• The scheduler requests a safe pause  
• The model saves progress  
• A lightweight task executes  
• Training resumes from the last checkpoint  

This is cooperative, application-level scheduling —  
not GPU-level preemption.

---

## 🔍 Real-World Use Cases

• Deep learning experimentation on personal workstations  
• Running inference safely alongside training  
• Shared single-GPU research environments  
• Preventing crashes from accidental concurrent launches  
• Managing GPU workloads without cluster infrastructure  

---

## 🏆 Why This Matters

Cluster schedulers solve GPU scheduling at data center scale.

GPU Scheduler brings structured workload control to:

• Developer laptops  
• Research workstations  
• Small labs  
• Personal ML setups  

It applies operating systems principles to GPU resource management in a lightweight, practical way.

---

## 🛠 Future Directions

Adaptive utilization tuning  
Predictive resource modeling  
Monitoring dashboard  
Multi-user fairness policies  
Container-aware scheduling  
Integration with ML experiment pipelines  

---

## 📚 Concepts Demonstrated

Operating Systems Scheduling  
Admission Control Algorithms  
Daemon-Based Monitoring  
GPU Resource Management  
Performance Stability Engineering  
Systems Architecture Design  

---
