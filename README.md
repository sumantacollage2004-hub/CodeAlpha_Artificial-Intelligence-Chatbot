https://sumantacollage2004-hub.github.io/CodeAlpha_Artificial-Intelligence-Chatbot/
# NLP AI Chatbot

A Java-architecture-inspired AI Chatbot built with HTML, CSS, and JavaScript.

## Files
- `index.html` — Main UI structure
- `style.css`  — External stylesheet (dark/light themes, responsive)
- `app.js`     — NLP engine, FAQ rules, Anthropic API integration

## NLP Pipeline (mirrors Java NLP libraries)
1. **Tokenizer** — splits input into tokens (like OpenNLP Tokenizer)
2. **Stop-word removal** — filters common words
3. **Stemmer** — Porter Stemmer (simplified, like Lucene/Snowball)
4. **Intent Classifier** — TF-IDF + Naive Bayes scoring
5. **NER** — regex-based Named Entity Recognition
6. **Sentiment Analysis** — lexicon-based (like VADER)

## Answer Layers
1. **Rule-based FAQ** — instant pattern-matched answers (HashMap-style)
2. **AI API** — Anthropic Claude for complex queries

## Features
- Real-time NLP analysis panel (intent, entities, sentiment, tokens)
- Dark/light theme toggle
- Session history sidebar
- Markdown rendering in responses
- Auto-resizing input
- Mobile responsive

## Usage
Open `index.html` in a browser. No build step required.
The Anthropic API key is handled by the hosting environment.
