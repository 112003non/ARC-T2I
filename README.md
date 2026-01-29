# ARC-T2I# ARC-T2I: Asynchronous Renewal-based Control for T2I Serving

This repository contains the simulator and implementation for the paper:

**ARC: Asynchronous Renewal-based Control for Cost-Latency Trade-offs in Text-to-Image Serving**  
(ICML 2026 submission)

## Overview

ARC is a queue-aware hierarchical control framework for text-to-image (T2I)
diffusion serving. It jointly optimizes:
- **Routing** across heterogeneous GPUs
- **Dispatch-time batching** via adaptive waiting times

ARC provides a theoretical **O(1/V)** optimality gap with **O(V)** queue backlog
and demonstrates improved latency–cost trade-offs in simulation.

## Repository Structure

- `arc/` — Core ARC routing and batching logic
- `simulator/` — Event-driven GPU queue simulator (SimPy-based)
- `experiments/` — Scripts for reproducing figures and tables
- `traces/` — Example arrival traces (DiffusionDB-based)
- `scripts/` — Trace analysis and plotting utilities

## Installation

```bash
git clone https://github.com/112003non/ARC-T2I.git
cd ARC-T2I
pip install -r requirements.txt
