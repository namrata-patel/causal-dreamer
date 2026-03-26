# Quick Reference: GitHub Repository Setup

## Folder Structure Visualization

```
📦 causal-dreamer/
│
├── 📄 index.html                    ← WEBSITE (must be in root!)
├── 📄 paper.pdf                     ← Your research paper
├── 📄 README.md                     ← Project description
│
├── 📁 images/                       ← All website images here
│   ├── 🖼️ model_architecture4.png
│   ├── 🖼️ result1.png
│   └── 🖼️ additional_results.png
│
├── 📁 src/                          ← Your Python code
│   ├── 📁 models/
│   ├── 📁 data/
│   ├── 📁 utils/
│   └── 📄 train.py
│
├── 📁 datasets/                     ← CausalLite10K dataset
│   └── 📄 causallite10k.json
│
├── 📁 checkpoints/                  ← Model weights
│   └── 📄 causal_adapter.pth
│
└── 📄 .gitignore                    ← Git ignore file
```

## Critical Rules ⚠️

1. **`index.html` MUST be in the ROOT directory** - This is the #1 mistake!
2. **Images go in `images/` folder** - Not in root, not in a subfolder of images
3. **Enable GitHub Pages** - Settings → Pages → Source: main branch, / (root)

## Three Commands to Deploy

```bash
git add .
git commit -m "Add GitHub Pages website"
git push origin main
```

Then wait 2-5 minutes and visit: `https://namrata-patel.github.io/causal-dreamer/`

## Common Mistakes to Avoid

❌ **DON'T DO THIS:**
```
causal-dreamer/
└── docs/
    └── index.html          ← Wrong! Not in root!
```

❌ **DON'T DO THIS:**
```
causal-dreamer/
└── website/
    └── index.html          ← Wrong! Not in root!
```

✅ **DO THIS:**
```
causal-dreamer/
├── index.html              ← Correct! In root!
└── images/
    └── result1.png
```

## URLs and Links

| Item | URL |
|------|-----|
| Website | `https://namrata-patel.github.io/causal-dreamer/` |
| Repository | `https://github.com/namrata-patel/causal-dreamer` |
| Paper PDF | `https://namrata-patel.github.io/causal-dreamer/paper.pdf` |

## What Each File Does

| File | Purpose |
|------|---------|
| `index.html` | The main website - contains all HTML, CSS, and JavaScript |
| `images/model_architecture4.png` | Architecture diagram (Figure 1) |
| `images/result1.png` | Comparison results (Figure 2) |
| `images/additional_results.png` | More examples (Figure 3) |
| `paper.pdf` | Full research paper for download |

## Testing Locally (Optional)

If you want to test the website on your computer before pushing:

```bash
# Option 1: Python 3
python -m http.server 8000

# Option 2: Python 2
python -m SimpleHTTPServer 8000

# Then visit: http://localhost:8000
```

## Mobile Responsive Breakpoints

The website is designed to work on:
- 📱 **Mobile**: < 768px
- 💻 **Tablet**: 768px - 1024px  
- 🖥️ **Desktop**: > 1024px

All layouts adapt automatically thanks to Bootstrap!

## Color Scheme

- **Primary**: Purple gradient (#667eea → #764ba2)
- **Text**: Dark blue (#2c3e50)
- **Backgrounds**: White and light gray (#f8f9fa)
- **Accents**: Blue (#3498db) and Red (#e74c3c)

---

**Ready to deploy?** Follow the steps in GITHUB_SETUP.md!
