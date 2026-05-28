/* =============================================
   NLP AI CHATBOT — app.js
   Java-style NLP Chatbot Architecture in JS
   ============================================= */

"use strict";

// ─────────────────────────────────────────────
// NLP ENGINE  (simulates Java NLP pipeline)
// ─────────────────────────────────────────────
const NLPEngine = (() => {

  // Tokenizer
  function tokenize(text) {
    return text.toLowerCase()
      .replace(/[^\w\s']/g, ' ')
      .split(/\s+/)
      .filter(Boolean);
  }

  // Stop-word removal (Java-style preprocessing)
  const STOP_WORDS = new Set([
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'may','might','shall','can','need','dare','ought','used','to','of',
    'in','on','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','from','up','down',
    'that','this','these','those','i','me','my','we','our','you','your',
    'he','she','it','they','them','what','which','who','when','where','how'
  ]);

  function removeStopWords(tokens) {
    return tokens.filter(t => !STOP_WORDS.has(t) && t.length > 1);
  }

  // Stemmer (Porter Stemmer simplified — mirrors Java NLP libraries)
  function stem(word) {
    if (word.length < 4) return word;
    const rules = [
      [/ational$/, 'ate'], [/tional$/, 'tion'], [/enci$/, 'ence'],
      [/anci$/, 'ance'], [/izer$/, 'ize'],  [/ising$/, 'ise'],
      [/izing$/, 'ize'], [/ness$/, ''],      [/ment$/, ''],
      [/ful$/, ''],       [/ous$/, ''],       [/ive$/, ''],
      [/ing$/, ''],       [/tion$/, 'te'],    [/ies$/, 'y'],
      [/sses$/, 'ss'],    [/es$/, ''],        [/s$/, '']
    ];
    for (const [pattern, replacement] of rules) {
      if (pattern.test(word)) return word.replace(pattern, replacement);
    }
    return word;
  }

  // TF-IDF style scoring for intent matching
  function tfScore(tokens, keywords) {
    const stemmed = tokens.map(stem);
    const kwStemmed = keywords.map(stem);
    return kwStemmed.reduce((score, kw) => {
      const matches = stemmed.filter(t => t === kw || t.includes(kw) || kw.includes(t)).length;
      return score + matches;
    }, 0);
  }

  // Named Entity Recognition (rule-based)
  function extractEntities(text) {
    const entities = [];
    const patterns = [
      { type: 'TECHNOLOGY', regex: /\b(java|python|javascript|nlp|ai|ml|deep learning|neural network|gpt|llm|bert|transformer|api|algorithm|machine learning|chatbot)\b/gi },
      { type: 'CONCEPT',    regex: /\b(accuracy|precision|recall|f1|bias|variance|overfitting|gradient|backpropagation|regression|classification|clustering)\b/gi },
      { type: 'LANGUAGE',   regex: /\b(english|spanish|french|german|chinese|hindi|arabic|japanese|natural language)\b/gi },
    ];
    for (const { type, regex } of patterns) {
      const found = [...new Set((text.match(regex) || []).map(e => e.toLowerCase()))];
      entities.push(...found.map(value => ({ type, value })));
    }
    return entities;
  }

  // Sentiment Analysis (lexicon-based — mirrors VADER / Stanford Sentiment)
  const POSITIVE_WORDS = new Set(['good','great','excellent','amazing','awesome','love','best','perfect','helpful','nice','wonderful','fantastic','brilliant','easy','fast','powerful','clear','smart','interesting','useful']);
  const NEGATIVE_WORDS = new Set(['bad','terrible','horrible','hate','worst','slow','confusing','difficult','wrong','broken','useless','poor','awful','ugly','stupid','boring','hard','annoying','error','fail']);

  function analyzeSentiment(tokens) {
    let score = 0;
    tokens.forEach(t => {
      if (POSITIVE_WORDS.has(t)) score++;
      if (NEGATIVE_WORDS.has(t)) score--;
    });
    if (score > 0) return { label: 'Positive 😊', value: score };
    if (score < 0) return { label: 'Negative 😟', value: score };
    return { label: 'Neutral 😐', value: 0 };
  }

  // Intent Classification (Naive Bayes-style scoring)
  const INTENTS = [
    { name: 'greeting',    keywords: ['hello','hi','hey','greetings','howdy','morning','evening','good day','sup','what up'] },
    { name: 'farewell',    keywords: ['bye','goodbye','see you','take care','later','farewell','cya','night'] },
    { name: 'thanks',      keywords: ['thanks','thank','appreciate','grateful','cheers','thx'] },
    { name: 'help',        keywords: ['help','assist','support','guide','explain','how','what','tell','show','teach'] },
    { name: 'definition',  keywords: ['what is','define','meaning','definition','describe','explain','overview','introduction'] },
    { name: 'comparison',  keywords: ['compare','difference','versus','vs','better','worse','pros','cons','advantages','disadvantages'] },
    { name: 'example',     keywords: ['example','sample','instance','demo','show me','give me','illustrate'] },
    { name: 'process',     keywords: ['how to','steps','process','procedure','workflow','implement','build','create','make','train'] },
    { name: 'opinion',     keywords: ['think','opinion','suggest','recommend','best','should','prefer','advice'] },
    { name: 'unknown',     keywords: [] }
  ];

  function classifyIntent(tokens) {
    const meaningful = removeStopWords(tokens);
    let best = { name: 'unknown', score: 0 };
    for (const intent of INTENTS) {
      const score = tfScore(meaningful, intent.keywords);
      if (score > best.score) best = { name: intent.name, score };
    }
    const confidence = Math.min(95, 45 + best.score * 12);
    return { intent: best.name, confidence };
  }

  // Full analysis pipeline
  function analyze(text) {
    const tokens    = tokenize(text);
    const filtered  = removeStopWords(tokens);
    const stems     = filtered.map(stem);
    const entities  = extractEntities(text);
    const sentiment = analyzeSentiment(filtered);
    const { intent, confidence } = classifyIntent(tokens);
    return { tokens, filtered, stems, entities, sentiment, intent, confidence };
  }

  return { analyze, tokenize, extractEntities, analyzeSentiment, classifyIntent };
})();


// ─────────────────────────────────────────────
// RULE-BASED FAQ KNOWLEDGE BASE
// Mirrors a Java HashMap / decision tree
// ─────────────────────────────────────────────
const FAQ = (() => {
  const rules = [
    {
      patterns: [/\bhello\b|\bhi\b|\bhey\b|\bgreet/i],
      response: "Hello! I'm your NLP-powered AI Assistant. I understand natural language, detect your intent, and analyze sentiment in real time. What would you like to explore today?",
      type: 'greeting'
    },
    {
      patterns: [/\bbye\b|\bgoodbye\b|\bsee you\b|\bfarewell/i],
      response: "Goodbye! It was great chatting with you. Come back anytime you have questions — I'm always here! 👋",
      type: 'farewell'
    },
    {
      patterns: [/\bthank/i],
      response: "You're very welcome! Is there anything else I can help you with?",
      type: 'thanks'
    },
    {
      patterns: [/what (is|are) (you|your name|this bot|this chatbot)/i, /who are you/i],
      response: "I'm an AI Chatbot built with **NLP (Natural Language Processing)** techniques. I perform tokenization, stop-word removal, stemming, intent classification, entity recognition, and sentiment analysis on every message you send — all in real time.",
      type: 'identity'
    },
    {
      patterns: [/what (is|are) nlp|natural language processing/i, /explain nlp/i],
      response: "**Natural Language Processing (NLP)** is a branch of AI that enables computers to understand, interpret, and generate human language.\n\nKey NLP tasks include:\n- **Tokenization** — splitting text into words/tokens\n- **POS Tagging** — labeling parts of speech\n- **NER** — identifying named entities (people, places, orgs)\n- **Sentiment Analysis** — detecting emotion polarity\n- **Intent Classification** — understanding user goals\n- **Machine Translation** — converting between languages\n\nPopular NLP libraries include NLTK, spaCy (Python) and Stanford CoreNLP, OpenNLP (Java).",
      type: 'nlp_definition'
    },
    {
      patterns: [/how does machine learning work|what is machine learning|explain ml/i],
      response: "**Machine Learning (ML)** is a subset of AI where systems learn patterns from data without being explicitly programmed.\n\n**Core approaches:**\n- **Supervised Learning** — learns from labeled input/output pairs (e.g. classification, regression)\n- **Unsupervised Learning** — finds hidden patterns without labels (e.g. clustering)\n- **Reinforcement Learning** — learns via reward signals (e.g. game-playing agents)\n\n**Workflow:** Collect data → Preprocess → Choose model → Train → Evaluate → Deploy",
      type: 'ml_definition'
    },
    {
      patterns: [/deep learning|neural network/i],
      response: "**Deep Learning** uses multi-layered artificial neural networks inspired by the human brain.\n\n**Key architectures:**\n- **CNNs** — image recognition\n- **RNNs/LSTMs** — sequential data, language\n- **Transformers** — state-of-the-art NLP (GPT, BERT)\n- **GANs** — generative models\n\nDeep learning powers speech recognition, computer vision, and large language models like the one running in this chatbot!",
      type: 'deep_learning'
    },
    {
      patterns: [/java.*chatbot|chatbot.*java|build.*chatbot/i],
      response: "To build a Java-based chatbot:\n\n1. **NLP Library** — Use Stanford CoreNLP or OpenNLP for tokenization, POS tagging, NER\n2. **Intent Engine** — Implement a Naive Bayes classifier or pattern matching with `java.util.regex`\n3. **Knowledge Base** — Use `HashMap<String, String>` for FAQ rules\n4. **ML Model** — Integrate Weka or Deeplearning4j for trained models\n5. **GUI/Web** — JavaFX for desktop, or Spring Boot + REST for web\n\nThis frontend is the web interface layer communicating with such a Java backend!",
      type: 'java_chatbot'
    },
    {
      patterns: [/what can you (do|help|assist)|your capabilities|features/i],
      response: "I can help you with:\n\n- 🧠 **NLP concepts** — tokenization, stemming, tagging, parsing\n- 🤖 **Machine learning** — algorithms, models, workflows\n- 💡 **AI architectures** — CNNs, RNNs, Transformers, LLMs\n- 📖 **Explanations** — any technical or general topic\n- 💬 **Conversation** — I analyze intent, entities & sentiment in real time\n\nJust type your question naturally!",
      type: 'capabilities'
    }
  ];

  function match(text) {
    for (const rule of rules) {
      if (rule.patterns.some(p => p.test(text))) {
        return rule.response;
      }
    }
    return null; // Fall through to AI API
  }

  return { match };
})();


// ─────────────────────────────────────────────
// ANTHROPIC API CLIENT
// ─────────────────────────────────────────────
async function callAnthropicAPI(messages) {
  const systemPrompt = `You are an expert AI assistant specializing in NLP, machine learning, deep learning, and Java programming.
You are running inside an NLP Chatbot application that showcases:
- Tokenization, stemming, stop-word removal
- Intent classification (Naive Bayes-style scoring)
- Named Entity Recognition (rule-based)
- Sentiment analysis (lexicon-based)
- Rule-based FAQ answers

When answering:
- Be precise, educational, and engaging
- Use markdown formatting (bold, lists, code blocks) when helpful
- Keep responses concise but complete
- If asked about the chatbot's architecture, explain the NLP pipeline
- For code examples, prefer Java or JavaScript`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: systemPrompt,
      messages: messages.map(m => ({ role: m.role, content: m.content }))
    })
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error?.message || `API error ${response.status}`);
  }

  const data = await response.json();
  return data.content.map(b => b.text || '').join('').trim();
}


// ─────────────────────────────────────────────
// CHAT STATE
// ─────────────────────────────────────────────
const State = {
  history: [],      // [{role, content}]
  sessions: [],
  currentSession: null,
  isLoading: false,
};


// ─────────────────────────────────────────────
// DOM REFERENCES
// ─────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dom = {
  sidebar:     $('sidebar'),
  sidebarToggle: $('sidebarToggle'),
  mobileMenuBtn: $('mobileMenuBtn'),
  newChatBtn:  $('newChatBtn'),
  chatHistory: $('chatHistory'),
  welcomeScreen: $('welcomeScreen'),
  messagesContainer: $('messagesContainer'),
  typingIndicator: $('typingIndicator'),
  userInput:   $('userInput'),
  sendBtn:     $('sendBtn'),
  clearBtn:    $('clearBtn'),
  themeBtn:    $('themeBtn'),
  nlpPanel:    $('nlpPanel'),
  nlpPanelBody: $('nlpPanelBody'),
};


// ─────────────────────────────────────────────
// RENDER HELPERS
// ─────────────────────────────────────────────
function markdownToHTML(text) {
  return text
    .replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang}">${escapeHTML(code.trim())}</code></pre>`)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
    .replace(/\n\n+/g, '</p><p>')
    .replace(/^(?!<[hupol])/gm, '')
    .replace(/\n/g, '<br>')
    .replace(/^(.+)/, p => p.startsWith('<') ? p : `<p>${p}</p>`);
}

function escapeHTML(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function timeNow() {
  return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function renderNLPTags(analysis) {
  const tags = [];
  tags.push(`<span class="nlp-tag intent" title="Detected intent">🎯 ${analysis.intent}</span>`);
  tags.push(`<span class="nlp-tag sentiment" title="Sentiment">${analysis.sentiment.label}</span>`);
  if (analysis.entities.length > 0) {
    const topEntity = analysis.entities[0];
    tags.push(`<span class="nlp-tag entity" title="Named entity">📌 ${topEntity.value}</span>`);
  }
  return `<div class="nlp-tags">${tags.join('')}</div>`;
}

function addMessage({ role, content, analysis }) {
  dom.welcomeScreen.style.display = 'none';

  const isBot = role === 'assistant';
  const msgEl = document.createElement('div');
  msgEl.className = `message ${isBot ? 'bot' : 'user'}`;

  const avatarHTML = isBot
    ? `<div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>`
    : `<div class="msg-avatar">U</div>`;

  const nlpHTML = isBot && analysis ? renderNLPTags(analysis) : '';
  const htmlContent = markdownToHTML(content);

  const copyBtn = isBot
    ? `<button class="msg-action-btn" onclick="copyText(this, \`${content.replace(/`/g, '\\`')}\`)"><i class="fa-regular fa-copy"></i> Copy</button>`
    : '';

  msgEl.innerHTML = `
    ${avatarHTML}
    <div class="msg-content">
      <div class="msg-bubble">${htmlContent}</div>
      ${nlpHTML}
      <div class="msg-meta">
        <span>${timeNow()}</span>
        ${isBot ? `<span>·</span><span>${analysis ? Math.round(analysis.confidence) + '% confidence' : ''}</span>` : ''}
      </div>
      <div class="msg-actions">${copyBtn}</div>
    </div>`;

  dom.messagesContainer.appendChild(msgEl);
  scrollToBottom();
}

function scrollToBottom() {
  dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
}

window.copyText = async function(btn, text) {
  await navigator.clipboard.writeText(text).catch(() => {});
  btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
  setTimeout(() => { btn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy'; }, 1500);
};


// ─────────────────────────────────────────────
// NLP PANEL
// ─────────────────────────────────────────────
function showNLPPanel(analysis, userText) {
  const { intent, confidence, entities, sentiment, tokens, filtered } = analysis;

  dom.nlpPanelBody.innerHTML = `
    <div class="nlp-stat">
      <div class="nlp-stat-label">Input text</div>
      <div class="nlp-stat-value" style="font-style:italic;color:var(--text-secondary)">"${escapeHTML(userText.slice(0,80))}${userText.length > 80 ? '…' : ''}"</div>
    </div>
    <div class="nlp-stat">
      <div class="nlp-stat-label">Detected Intent</div>
      <div class="nlp-stat-value"><strong>${intent}</strong></div>
      <div class="confidence-bar"><div class="confidence-fill" style="width:${confidence}%"></div></div>
      <div style="font-size:12px;color:var(--text-muted);margin-top:4px">${Math.round(confidence)}% confidence</div>
    </div>
    <div class="nlp-stat">
      <div class="nlp-stat-label">Sentiment</div>
      <div class="nlp-stat-value">${sentiment.label} (score: ${sentiment.value > 0 ? '+' : ''}${sentiment.value})</div>
    </div>
    <div class="nlp-stat">
      <div class="nlp-stat-label">Tokens (${tokens.length})</div>
      <div class="nlp-stat-value" style="font-size:13px;color:var(--text-secondary)">${tokens.slice(0,12).join(', ')}${tokens.length > 12 ? '…' : ''}</div>
    </div>
    <div class="nlp-stat">
      <div class="nlp-stat-label">After stop-word removal (${filtered.length})</div>
      <div class="nlp-stat-value" style="font-size:13px;color:var(--text-secondary)">${filtered.join(', ') || '—'}</div>
    </div>
    <div class="nlp-stat">
      <div class="nlp-stat-label">Named Entities (${entities.length})</div>
      <div class="nlp-stat-value" style="font-size:13px">
        ${entities.length === 0 ? '<span style="color:var(--text-muted)">None detected</span>' :
          entities.map(e => `<span class="nlp-tag entity" style="display:inline-block;margin:2px">${e.value} <span style="opacity:0.6">(${e.type})</span></span>`).join(' ')}
      </div>
    </div>
    <div class="nlp-stat">
      <div class="nlp-stat-label">Answer Source</div>
      <div class="nlp-stat-value" style="color:var(--accent-light)">${analysis.source === 'faq' ? '📚 FAQ / Rule-based' : '🤖 AI Language Model'}</div>
    </div>`;

  dom.nlpPanel.hidden = false;
}


// ─────────────────────────────────────────────
// MAIN SEND LOGIC
// ─────────────────────────────────────────────
async function sendMessage(text) {
  text = text.trim();
  if (!text || State.isLoading) return;

  State.isLoading = true;
  dom.sendBtn.disabled = true;
  dom.userInput.value = '';
  dom.userInput.style.height = 'auto';

  // Run NLP pipeline on user input
  const analysis = NLPEngine.analyze(text);

  // Add user message
  addMessage({ role: 'user', content: text });
  State.history.push({ role: 'user', content: text });

  // Show typing
  dom.typingIndicator.hidden = false;
  scrollToBottom();

  try {
    let botReply;
    let source = 'ai';

    // 1. Try rule-based FAQ first
    const faqReply = FAQ.match(text);
    if (faqReply) {
      botReply = faqReply;
      source = 'faq';
    } else {
      // 2. Fall through to Anthropic AI API
      botReply = await callAnthropicAPI(State.history);
    }

    analysis.source = source;

    dom.typingIndicator.hidden = true;
    addMessage({ role: 'assistant', content: botReply, analysis });
    State.history.push({ role: 'assistant', content: botReply });

    // Update sidebar session
    updateSessionHistory(text);

    // Auto-show NLP panel for every AI response
    showNLPPanel(analysis, text);

  } catch (err) {
    dom.typingIndicator.hidden = true;
    addMessage({
      role: 'assistant',
      content: `⚠️ **Connection error:** ${err.message}\n\nPlease check your connection and try again.`,
      analysis: { ...analysis, source: 'error' }
    });
  }

  State.isLoading = false;
  dom.sendBtn.disabled = false;
  dom.userInput.focus();
}

window.sendSuggestion = function(text) {
  dom.userInput.value = text;
  sendMessage(text);
};


// ─────────────────────────────────────────────
// SESSION MANAGEMENT
// ─────────────────────────────────────────────
function updateSessionHistory(firstMessage) {
  if (State.currentSession) return;
  State.currentSession = { id: Date.now(), title: firstMessage.slice(0, 30) };
  State.sessions.unshift(State.currentSession);
  renderSessions();
}

function renderSessions() {
  const existing = dom.chatHistory.querySelectorAll('.chat-history-item');
  existing.forEach(e => e.remove());

  const label = dom.chatHistory.querySelector('.history-label');

  State.sessions.slice(0, 8).forEach(session => {
    const item = document.createElement('div');
    item.className = 'chat-history-item' + (session.id === State.currentSession?.id ? ' active' : '');
    item.innerHTML = `<i class="fa-regular fa-message"></i><span>${session.title}…</span>`;
    label.after(item);
  });
}

function startNewChat() {
  State.history = [];
  State.currentSession = null;
  dom.messagesContainer.innerHTML = '';
  dom.welcomeScreen.style.display = '';
  dom.nlpPanel.hidden = true;
  renderSessions();
  dom.userInput.focus();
}


// ─────────────────────────────────────────────
// EVENT LISTENERS
// ─────────────────────────────────────────────
dom.sendBtn.addEventListener('click', () => sendMessage(dom.userInput.value));

dom.userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(dom.userInput.value);
  }
});

// Auto-resize textarea
dom.userInput.addEventListener('input', () => {
  dom.userInput.style.height = 'auto';
  dom.userInput.style.height = Math.min(dom.userInput.scrollHeight, 140) + 'px';
});

dom.newChatBtn.addEventListener('click', startNewChat);
dom.clearBtn.addEventListener('click', startNewChat);

// Sidebar toggle
dom.sidebarToggle.addEventListener('click', () => {
  dom.sidebar.classList.toggle('collapsed');
});

// Mobile menu
dom.mobileMenuBtn.addEventListener('click', () => {
  dom.sidebar.classList.toggle('mobile-open');
});

// Theme toggle
let isDark = true;
dom.themeBtn.addEventListener('click', () => {
  isDark = !isDark;
  document.body.classList.toggle('light', !isDark);
  dom.themeBtn.querySelector('i').className = isDark ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
});
