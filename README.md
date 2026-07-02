# ScoreMe MSME Pipeline Scheduling

## Overview
This repository contains my submission for the **ScoreMe Advanced Systems Design Coding Assignment**. 
It implements a hybrid heuristic algorithm for the MSME Credit Pipeline Scheduling problem, combining **DSATUR** (graph coloring) with **Tabu Local Search** to minimize weighted delay and load imbalance.

## Problem Constraints Handled
- **F1:** Conflict avoidance (Graph Coloring)
- **F2:** Resource capacity (CPU, RAM, GPU, Network)
- **F3:** SLA Time Windows

## Files Included
- `scheduler.py`: Main scheduling algorithm implementation.
- `generator.py`: Random instance generator for benchmarking.
- `ScoreMe_Report.docx`: Complete assignment report (Tasks 1-7) with empirical analysis.
- `charts.png`: Visualization of penalty vs. number of tasks, and runtime scaling.

## How to Run
1. Generate an instance:
   ```bash
   python -c "import generator, json; json.dump(generator.generate_instance(8, 3, 0.3, 1), open('inst.json','w'))"
