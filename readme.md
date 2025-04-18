# 🧠 Quiz RAG: Trivia Knowledge System

An AI- and ML-powered trivia learning system. Where most quiz archives use string search, this app is built on semantic search. Furthermore, it uses Claude to synthesis the query results and provide additional context.

## Features

- **Semantic Search**: Find relevant trivia questions using natural language queries
- **AI-Powered Synthesis**: Generate educational summaries from related trivia facts using Claude AI
- **Dual Synthesis Modes**:
  - Basic Mode: Concise summaries using Claude Haiku
  - World Knowledge Mode: In-depth explanations with Claude Sonnet, incorporating broader context
- **Interactive UI**: User-friendly Streamlit interface with configurable settings
- **Smart Ranking**: Questions ranked by semantic similarity to your query

## Usage

1. **Search**: Enter any topic or question in the search bar
2. **Configure**:
   - Adjust number of results (5-25)
   - Enable/disable AI synthesis
   - Toggle world knowledge integration
3. **Explore**: 
   - View matching questions with similarity scores
   - Read AI-generated knowledge synthesis

### Search Process

1. Questions, answers, and categories are embedded using Sentence Transformers
2. Cosine similarity is used to find the most relevant trivia items
