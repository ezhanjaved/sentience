# Sentience

### *An Embodied AI Framework for Hierarchical LLM-RL Agents*

> **Sentience** explores how Large Language Models (LLMs) and Reinforcement Learning (RL) can work together to build autonomous embodied agents. Instead of replacing RL with an LLM, Sentience separates **reasoning** from **execution**: the LLM plans and makes strategic decisions while specialized RL policies perform low-level motor control inside the Craftax environment.

---

## Vision

Recent advances in Large Language Models have demonstrated impressive reasoning capabilities, while Reinforcement Learning remains one of the strongest approaches for learning continuous control and embodied behaviors.

Sentience combines both paradigms into a modular architecture where:

* 🧠 **LLMs** reason, plan, remember, and select goals.
* 🤖 **RL agents** execute specialized motor skills.
* 🧩 **LangGraph** orchestrates decision workflows.
* 📚 **RAG & Vector Databases** provide long-term memory and knowledge retrieval.
* 🌍 **Craftax** serves as the embodied environment for experimentation.

The objective is to investigate whether hierarchical AI systems outperform monolithic agents on long-horizon survival tasks.

---

# Architecture

```
                 Craftax Environment
                        │
                        ▼
               World Observation
                        │
                        ▼
                 LangGraph Agent
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     Tool Calls      RAG Memory     World State
        │               │
        └───────────────┘
                ▼
          LLM Planner
                ▼
        Skill Selection
                ▼
      Reinforcement Learning
        ├── Navigate
        ├── Gather
        ├── Fight
        ├── Escape
        ├── Craft
        └── Explore
                ▼
          Execute Actions
                ▼
             Craftax
```

---

# Project Goals

Sentience is designed to explore:

* Hierarchical AI architectures
* Embodied intelligence
* Long-horizon planning
* Autonomous decision making
* LLM tool usage
* Skill composition
* Episodic memory
* Retrieval-Augmented Generation (RAG)
* Multi-agent workflows (future)
* Human understandable reasoning

---

# Core Technologies

| Component              | Purpose                           |
| ---------------------- | --------------------------------- |
| Craftax                | Embodied survival environment     |
| LLM                    | High-level reasoning and planning |
| Reinforcement Learning | Low-level motor control           |
| LangChain              | Tool integration                  |
| LangGraph              | Agent workflow orchestration      |
| RAG                    | Knowledge retrieval               |
| Vector Database        | Long-term semantic memory         |
| Python                 | Backend                           |
| JAX                    | Craftax runtime                   |

---

# Design Philosophy

Unlike traditional game AI, Sentience separates intelligence into two distinct layers.

## High-Level Intelligence (LLM)

Responsible for:

* Understanding goals
* Planning
* Reasoning
* Tool usage
* Reflection
* Memory retrieval
* Skill selection

Example:

> "Night is approaching and food is low. Gather food before returning to shelter."

---

## Low-Level Intelligence (RL)

Responsible for:

* Navigation
* Resource gathering
* Combat
* Exploration
* Obstacle avoidance
* Motor execution

The RL policies never decide **what** to do.

They only decide **how** to do it.

---

# Planned Features

## Phase 1

* [ ] Learn Craftax API
* [ ] Environment wrapper
* [ ] Observation parser
* [ ] Random agent

---

## Phase 2

* [ ] Environment tools
* [ ] Inventory inspection
* [ ] Resource detection
* [ ] Health monitoring
* [ ] Goal interface

---

## Phase 3

* [ ] LangChain integration
* [ ] Tool calling
* [ ] Structured outputs
* [ ] Goal planner

---

## Phase 4

* [ ] Reinforcement Learning skills

* [ ] Navigation

* [ ] Gathering

* [ ] Exploration

* [ ] Combat

* [ ] Escape

---

## Phase 5

* [ ] LangGraph workflow

```
Observe

↓

Reason

↓

Choose Skill

↓

Execute

↓

Evaluate

↓

Succeeded?

↓

Yes → Continue

↓

No → Replan
```

---

## Phase 6

* [ ] Memory
* [ ] Embeddings
* [ ] Vector Database
* [ ] Retrieval-Augmented Generation (RAG)

---

## Phase 7

* [ ] Reflection
* [ ] Failure analysis
* [ ] Dynamic replanning

---

## Phase 8

* [ ] Multi-agent coordination
* [ ] Agent communication
* [ ] Cooperative survival

---

# Repository Structure

```
Sentience/

├── planner/
│   ├── langgraph/
│   ├── prompts/
│   ├── memory/
│   └── reasoning/
│
├── rl/
│   ├── navigation/
│   ├── gather/
│   ├── fight/
│   ├── explore/
│   └── escape/
│
├── tools/
│   ├── inventory.py
│   ├── observations.py
│   ├── skills.py
│   └── environment.py
│
├── craftax/
│
├── evaluation/
│
├── experiments/
│
├── docs/
│
└── README.md
```

---

# Long-Term Research Questions

Sentience aims to explore questions such as:

* Can LLM planning improve long-horizon survival?
* How many reusable RL skills are sufficient for complex tasks?
* Does episodic memory improve decision quality?
* Can reflection reduce repeated failures?
* When should an LLM switch between RL skills?
* Can hierarchical agents outperform end-to-end RL?

---

# Why Sentience?

Most current AI systems excel at either **reasoning** or **execution**.

Sentience explores the intersection of both.

Rather than expecting an LLM to directly control an embodied agent, the project investigates a hierarchical architecture where language models reason over long-term objectives while reinforcement learning policies provide reliable low-level execution.

The result is a modular, explainable, and extensible framework for building autonomous embodied AI agents.

---

## Status

> 🚧 **Early Development**

This project is currently focused on understanding the Craftax environment and building the core architecture before integrating LLM planning, memory systems, and reinforcement learning skills.

---
