# Pet Care Support Agent

Phase 1 — Data Pipeline (Alex) - 4/6/2026

Nothing else can be built until this works. This is your week one priority.

Step 1: Scraper

- Write a Python scraper targeting PetMD, ASPCA, AKC, VCA

- Output: raw text files, one per article

- Key constraint: respect robots.txt, add delays between requests

- Deliverable: scraper.py

Step 2: Cleaner & Structurer

- Parse raw text into consistent JSON schema

- Schema: {title, content, source, url, species, topic_category}

- Deliverable: cleaner.py + a folder of clean .json files

Step 3: Chunker

- Split each document into ~400 token chunks with ~50 token overlap

- Preserve metadata per chunk: {source, url, species, topic_category}

- Deliverable: chunker.py

Step 4: Embedder & Vector Store

- Embed chunks using sentence-transformers

- Store in ChromaDB with metadata

- Deliverable: vectorstore.py + populated local ChromaDB

Step 5: Retriever

- Clean interface: retrieve(query, top_k, filters)

- This is the handoff point — everything above this line is data engineering, everything below is agent logic

- Deliverable: retriever.py

Phase 2 — Agent Core (Peter, Alex) (Tuesday 4/7/2026)

Depends on retriever.py from Phase 1. Can sketch the logic before, but can't test it.

Step 6: Intent Classifier

- Few-shot prompt to Qwen3 classifying queries into: food_safety | symptom_triage | care_routine | general_qa | out_of_scope

- Deliverable: intent_classifier.py

Step 7: Agent Loop

- Receive query → classify intent → route to handler → return response with source attribution

- Deliverable: agent.py

Phase 3 — Actions & State (Peter, Angela, Emilia) (4/10/2026)

Depends on agent.py. This is the feedback's most critical gap.

Step 8: Food/Toxin Checker

- Single turn: species + food/substance → risk level + source

- Deliverable: actions/food_safety.py

Step 9: Symptom Triage

- Multi-turn: collect species → symptoms → duration → weight across turns

- State machine: track what's been collected, prompt for what's missing

- Deliverable: actions/symptom_triage.py

Step 10: Pet Profile & Care Routine

- Stateful across sessions: store pet name, species, age to disk

- Personalized advice on return visits

- Deliverable: actions/pet_profile.py + state/profiles.json

Phase 4 — Guardrails (Victor) (4/10/2026)

Can be built in parallel with Phase 3.

Step 11: Guardrails

- System prompt with domain restrictions

- Prompt injection detection

- "I don't know" behavior when RAG context is insufficient

- Medical disclaimer on all triage responses

- Deliverable: guardrails.py

Phase 5 — Evaluation (Andrew) (4/12/2026)

Write test cases before Phase 3. Run them after Phase 4.

Step 12: Test Suite

- 15 test cases covering: retrieval, routing, actions, out-of-scope rejection, prompt injection, error handling

- Run against full system, log pass/fail

- Write honest failure analysis

- Deliverable: evaluation/test_cases.py + evaluation/results.md

Phase 6 — UI & Presentation - 4/12/2026

Last. Don't touch this until Phase 4 is solid.

Step 13: Streamlit UI

- Chat interface, pet profile panel, source attribution display

- Wire to agent.py

- Deliverable: app.py

Step 14: README & Deck

- README: setup instructions, architecture diagram, dependencies

- Deck: 5 slides — problem, architecture, demo, evaluation results, limitations

- Backup demo video recorded while system is working

Folder Structure

This separation is what makes it portable to RacquetIQ later:

pet_care_agent/  
├── data/  
│   ├── scraper.py  
│   ├── cleaner.py  
│   └── chunker.py  
├── vectorstore.py  
├── retriever.py  
├── intent_classifier.py  
├── agent.py  
├── actions/  
│   ├── food_safety.py  
│   ├── symptom_triage.py  
│   └── pet_profile.py  
├── state/  
│   └── profiles.json  
├── guardrails.py  
├── evaluation/  
│   ├── test_cases.py  
│   └── results.md  
└── app.py**
