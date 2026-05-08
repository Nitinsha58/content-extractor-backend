"""
LaTeX Cleaner — built from the complete LaTeX Mathematical Symbols reference
All regex compiled once at import time. Just call clean_latex(expr).
To add a new rule: append one _r(...) tuple to the right section.
"""

import re

def _r(pattern, replacement, desc=""):
    """Compile pattern once."""
    return (re.compile(pattern), replacement, desc)


# ═══════════════════════════════════════════════════════════════
# 1. OCR ARTIFACT CLEANUP  (always run first)
# ═══════════════════════════════════════════════════════════════
CLEANUP_RULES = [
    _r(r'^```latex\s*',  '',  "strip opening latex fence"),
    _r(r'^```\s*',       '',  "strip opening fence"),
    _r(r'```\s*$',       '',  "strip closing fence"),
    _r(r'^\$\$\s*',      '',  "strip leading $$"),
    _r(r'\s*\$\$$',      '',  "strip trailing $$"),
    _r(r'^\$\s*',        '',  "strip leading $"),
    _r(r'\s*\$$',        '',  "strip trailing $"),
    _r(r'\\\\$',         '',  "strip trailing newline cmd"),
]

# ═══════════════════════════════════════════════════════════════
# 2. SPACING TOKENS  (ref: spacing commands)
# ═══════════════════════════════════════════════════════════════
SPACING_RULES = [
    _r(r'\\[,:;!]\s*',        ' ',  "thin/thick/neg space tokens"),
    _r(r'\\quad\s*',          ' ',  "quad"),
    _r(r'\\qquad\s*',         '  ', "qquad"),
    _r(r'\\hspace\{[^}]*\}',  ' ',  "hspace"),
    _r(r'\\vspace\{[^}]*\}',  '',   "vspace"),
    _r(r'~',                  ' ',  "non-breaking space"),
]

# ═══════════════════════════════════════════════════════════════
# 3. GREEK LETTERS  (ref: Section 1)
# ═══════════════════════════════════════════════════════════════
GREEK_RULES = [
    # Lowercase
    _r(r'α', r'\\alpha',      "α"), _r(r'β', r'\\beta',      "β"),
    _r(r'γ', r'\\gamma',      "γ"), _r(r'δ', r'\\delta',     "δ"),
    _r(r'ε', r'\\epsilon',    "ε"), _r(r'ζ', r'\\zeta',      "ζ"),
    _r(r'η', r'\\eta',        "η"), _r(r'θ', r'\\theta',     "θ"),
    _r(r'ι', r'\\iota',       "ι"), _r(r'κ', r'\\kappa',     "κ"),
    _r(r'λ', r'\\lambda',     "λ"), _r(r'μ', r'\\mu',        "μ"),
    _r(r'ν', r'\\nu',         "ν"), _r(r'ξ', r'\\xi',        "ξ"),
    _r(r'π', r'\\pi',         "π"), _r(r'ρ', r'\\rho',       "ρ"),
    _r(r'σ', r'\\sigma',      "σ"), _r(r'τ', r'\\tau',       "τ"),
    _r(r'υ', r'\\upsilon',    "υ"), _r(r'φ', r'\\phi',       "φ"),
    _r(r'χ', r'\\chi',        "χ"), _r(r'ψ', r'\\psi',       "ψ"),
    _r(r'ω', r'\\omega',      "ω"),
    # Variants (ref: varepsilon varkappa varphi varpi varrho varsigma vartheta digamma)
    _r(r'ϵ', r'\\varepsilon', "ϵ"), _r(r'ϑ', r'\\vartheta',  "ϑ"),
    _r(r'ϕ', r'\\varphi',    "ϕ"), _r(r'ϱ', r'\\varrho',    "ϱ"),
    _r(r'ϖ', r'\\varpi',     "ϖ"), _r(r'ϝ', r'\\digamma',   "ϝ"),
    _r(r'ϰ', r'\\varkappa',  "ϰ"), _r(r'ς', r'\\varsigma',  "ς"),
    # Uppercase
    _r(r'Γ', r'\\Gamma',     "Γ"), _r(r'Δ', r'\\Delta',     "Δ"),
    _r(r'Θ', r'\\Theta',     "Θ"), _r(r'Λ', r'\\Lambda',    "Λ"),
    _r(r'Ξ', r'\\Xi',        "Ξ"), _r(r'Π', r'\\Pi',        "Π"),
    _r(r'Σ', r'\\Sigma',     "Σ"), _r(r'Υ', r'\\Upsilon',   "Υ"),
    _r(r'Φ', r'\\Phi',       "Φ"), _r(r'Ψ', r'\\Psi',       "Ψ"),
    _r(r'Ω', r'\\Omega',     "Ω"),
    # Hebrew (ref: aleph beth daleth gimel)
    _r(r'ℵ', r'\\aleph',     "ℵ"), _r(r'ℶ', r'\\beth',      "ℶ"),
    _r(r'ℸ', r'\\daleth',    "ℸ"), _r(r'ℷ', r'\\gimel',     "ℷ"),
]

# ═══════════════════════════════════════════════════════════════
# 4. MATH CONSTRUCTS  (ref: Section 2)
# ═══════════════════════════════════════════════════════════════
CONSTRUCT_RULES = [
    _r(r'\\frac\s+\{',           r'\\frac{',       "space between frac and brace"),
    _r(r'\\frac\s+(\w)\s+(\w)',  r'\\frac{\1}{\2}',"bare frac args"),
    _r(r'\\sqrt\s+(\w)',         r'\\sqrt{\1}',    "bare sqrt arg"),
    _r(r'\^([A-Za-z0-9]{2,})',   r'^{\1}',         "bare multi-char superscript"),
    _r(r'_([A-Za-z0-9]{2,})',    r'_{\1}',         "bare multi-char subscript"),
    _r(r'_\{\s*\\,\s*',          '_{',             "spacing token in subscript"),
    _r(r'\\overline\s*\{',       r'\\overline{',   "overline spacing"),
    _r(r'\\underline\s*\{',      r'\\underline{',  "underline spacing"),
    _r(r'\\overbrace\s*\{',      r'\\overbrace{',  "overbrace spacing"),
    _r(r'\\underbrace\s*\{',     r'\\underbrace{', "underbrace spacing"),
    _r(r'\\widehat\s*\{',        r'\\widehat{',    "widehat spacing"),
    _r(r'\\widetilde\s*\{',      r'\\widetilde{',  "widetilde spacing"),
    _r(r'\\overrightarrow\s*\{', r'\\overrightarrow{', "overrightarrow spacing"),
    _r(r'\\overleftarrow\s*\{',  r'\\overleftarrow{',  "overleftarrow spacing"),
]

# ═══════════════════════════════════════════════════════════════
# 5. DELIMITERS  (ref: Section 3)
# ═══════════════════════════════════════════════════════════════
DELIMITER_RULES = [
    _r(r'\\left\s*\(',    r'\\left(',    r"\left ("),
    _r(r'\\right\s*\)',   r'\\right)',   r"\right )"),
    _r(r'\\left\s*\[',    r'\\left[',   r"\left ["),
    _r(r'\\right\s*\]',   r'\\right]',  r"\right ]"),
    _r(r'\\left\s*\\\{',  r'\\left\{',  r"\left {"),
    _r(r'\\right\s*\\\}', r'\\right\}', r"\right }"),
    _r(r'\{\s+',          '{',          "space after open brace"),
    _r(r'\s+\}',          '}',          "space before close brace"),
]

# ═══════════════════════════════════════════════════════════════
# 6. BINARY OPERATORS  (ref: Section 6)
# ═══════════════════════════════════════════════════════════════
BINARY_OP_RULES = [
    _r(r'×',  r'\\times',          "×"), _r(r'÷',  r'\\div',            "÷"),
    _r(r'±',  r'\\pm',             "±"), _r(r'∓',  r'\\mp',             "∓"),
    _r(r'·',  r'\\cdot',           "·"), _r(r'∘',  r'\\circ',           "∘"),
    _r(r'•',  r'\\bullet',         "•"), _r(r'⊕',  r'\\oplus',          "⊕"),
    _r(r'⊗',  r'\\otimes',         "⊗"), _r(r'⊖',  r'\\ominus',         "⊖"),
    _r(r'⊘',  r'\\oslash',         "⊘"), _r(r'⊙',  r'\\odot',           "⊙"),
    _r(r'†',  r'\\dagger',         "†"), _r(r'‡',  r'\\ddagger',        "‡"),
    _r(r'∧',  r'\\wedge',          "∧"), _r(r'∨',  r'\\vee',            "∨"),
    _r(r'∩',  r'\\cap',            "∩"), _r(r'∪',  r'\\cup',            "∪"),
    _r(r'⊔',  r'\\sqcup',          "⊔"), _r(r'⊓',  r'\\sqcap',          "⊓"),
    _r(r'⊎',  r'\\uplus',          "⊎"), _r(r'⊠',  r'\\boxtimes',       "⊠"),
    _r(r'⊞',  r'\\boxplus',        "⊞"), _r(r'⊟',  r'\\boxminus',       "⊟"),
    _r(r'⊡',  r'\\boxdot',         "⊡"), _r(r'⊻',  r'\\veebar',         "⊻"),
    _r(r'⊼',  r'\\barwedge',       "⊼"), _r(r'⋏',  r'\\curlywedge',     "⋏"),
    _r(r'⋎',  r'\\curlyvee',       "⋎"), _r(r'⊥',  r'\\bot',            "⊥"),
    _r(r'⊤',  r'\\top',            "⊤"), _r(r'⊺',  r'\\intercal',       "⊺"),
    _r(r'△',  r'\\bigtriangleup',  "△"), _r(r'▽',  r'\\bigtriangledown',"▽"),
    _r(r'◁',  r'\\triangleleft',   "◁"), _r(r'▷',  r'\\triangleright',  "▷"),
    # asterisk between identifiers → \cdot
    _r(r'(?<=[A-Za-z0-9}])\s*\*\s*(?=[A-Za-z0-9{\\])', r'\\cdot', "* to cdot"),
]

# ═══════════════════════════════════════════════════════════════
# 7. RELATION SYMBOLS  (ref: Section 6)
# ═══════════════════════════════════════════════════════════════
RELATION_RULES = [
    _r(r'≡',  r'\\equiv',          "≡"), _r(r'≅',  r'\\cong',           "≅"),
    _r(r'≈',  r'\\approx',         "≈"), _r(r'∼',  r'\\sim',            "∼"),
    _r(r'≃',  r'\\simeq',          "≃"), _r(r'≐',  r'\\doteq',          "≐"),
    _r(r'∝',  r'\\propto',         "∝"), _r(r'∣',  r'\\mid',            "∣"),
    _r(r'∥',  r'\\parallel',       "∥"), _r(r'⊂',  r'\\subset',         "⊂"),
    _r(r'⊃',  r'\\supset',         "⊃"), _r(r'⊆',  r'\\subseteq',       "⊆"),
    _r(r'⊇',  r'\\supseteq',       "⊇"), _r(r'⊏',  r'\\sqsubset',       "⊏"),
    _r(r'⊐',  r'\\sqsupset',       "⊐"), _r(r'⊑',  r'\\sqsubseteq',     "⊑"),
    _r(r'⊒',  r'\\sqsupseteq',     "⊒"), _r(r'∈',  r'\\in',             "∈"),
    _r(r'∉',  r'\\notin',          "∉"), _r(r'∋',  r'\\ni',             "∋"),
    _r(r'⊢',  r'\\vdash',          "⊢"), _r(r'⊣',  r'\\dashv',          "⊣"),
    _r(r'⊨',  r'\\models',         "⊨"), _r(r'≺',  r'\\prec',           "≺"),
    _r(r'≻',  r'\\succ',           "≻"), _r(r'≼',  r'\\preceq',         "≼"),
    _r(r'≽',  r'\\succeq',         "≽"), _r(r'≪',  r'\\ll',             "≪"),
    _r(r'≫',  r'\\gg',             "≫"), _r(r'⌣',  r'\\smile',          "⌣"),
    _r(r'⌢',  r'\\frown',          "⌢"), _r(r'∴',  r'\\therefore',      "∴"),
    _r(r'∵',  r'\\because',        "∵"), _r(r'≠',  r'\\neq',            "≠"),
    _r(r'≤',  r'\\leq',            "≤"), _r(r'≥',  r'\\geq',            "≥"),
    _r(r'≦',  r'\\leqq',           "≦"), _r(r'≧',  r'\\geqq',           "≧"),
    _r(r'⩽',  r'\\leqslant',       "⩽"), _r(r'⩾',  r'\\geqslant',       "⩾"),
    _r(r'≁',  r'\\nsim',           "≁"), _r(r'≇',  r'\\ncong',          "≇"),
    _r(r'∤',  r'\\nmid',           "∤"), _r(r'∦',  r'\\nparallel',      "∦"),
    _r(r'⊄',  r'\\nsubseteq',      "⊄"), _r(r'⊅',  r'\\nsupseteq',      "⊅"),
    _r(r'≮',  r'\\nless',          "≮"), _r(r'≯',  r'\\ngtr',           "≯"),
    _r(r'≰',  r'\\nleq',           "≰"), _r(r'≱',  r'\\ngeq',           "≱"),
    # ASCII comparison patterns from OCR
    _r(r'(?<![<\\])<=(?!=)',        r'\\leq',   "<= to leq"),
    _r(r'(?<![>\\])>=(?!=)',        r'\\geq',   ">= to geq"),
    _r(r'(?<![!=])!=',              r'\\neq',   "!= to neq"),
]

# ═══════════════════════════════════════════════════════════════
# 8. ARROW SYMBOLS  (ref: Section 7)
# ═══════════════════════════════════════════════════════════════
ARROW_RULES = [
    _r(r'←',  r'\\leftarrow',          "←"), _r(r'→',  r'\\rightarrow',         "→"),
    _r(r'↑',  r'\\uparrow',            "↑"), _r(r'↓',  r'\\downarrow',          "↓"),
    _r(r'↔',  r'\\leftrightarrow',     "↔"), _r(r'↕',  r'\\updownarrow',        "↕"),
    _r(r'⇐',  r'\\Leftarrow',          "⇐"), _r(r'⇒',  r'\\Rightarrow',         "⇒"),
    _r(r'⇑',  r'\\Uparrow',            "⇑"), _r(r'⇓',  r'\\Downarrow',          "⇓"),
    _r(r'⇔',  r'\\Leftrightarrow',     "⇔"), _r(r'⇕',  r'\\Updownarrow',        "⇕"),
    _r(r'↦',  r'\\mapsto',             "↦"), _r(r'↩',  r'\\hookleftarrow',      "↩"),
    _r(r'↪',  r'\\hookrightarrow',     "↪"), _r(r'↼',  r'\\leftharpoonup',      "↼"),
    _r(r'↽',  r'\\leftharpoondown',    "↽"), _r(r'⇀',  r'\\rightharpoonup',     "⇀"),
    _r(r'⇁',  r'\\rightharpoondown',   "⇁"), _r(r'⇌',  r'\\rightleftharpoons',  "⇌"),
    _r(r'↗',  r'\\nearrow',            "↗"), _r(r'↘',  r'\\searrow',            "↘"),
    _r(r'↙',  r'\\swarrow',            "↙"), _r(r'↖',  r'\\nwarrow',            "↖"),
    _r(r'↠',  r'\\twoheadrightarrow',  "↠"), _r(r'↞',  r'\\twoheadleftarrow',   "↞"),
    _r(r'↣',  r'\\rightarrowtail',     "↣"), _r(r'↢',  r'\\leftarrowtail',      "↢"),
    _r(r'↻',  r'\\circlearrowright',   "↻"), _r(r'↺',  r'\\circlearrowleft',    "↺"),
    _r(r'↚',  r'\\nleftarrow',         "↚"), _r(r'↛',  r'\\nrightarrow',        "↛"),
    _r(r'⇍',  r'\\nLeftarrow',         "⇍"), _r(r'⇏',  r'\\nRightarrow',        "⇏"),
    _r(r'⇎',  r'\\nLeftrightarrow',    "⇎"), _r(r'⇝',  r'\\leadsto',            "⇝"),
    # ASCII patterns from OCR — order: longest first
    _r(r'<=>',                          r'\\Leftrightarrow', "<=> pattern"),
    _r(r'(?<![=\-])-+>',               r'\\rightarrow',     "-> pattern"),
    _r(r'<-+(?![=\-])',                r'\\leftarrow',      "<- pattern"),
    _r(r'=+>',                          r'\\Rightarrow',     "=> pattern"),
]

# ═══════════════════════════════════════════════════════════════
# 9. VARIABLE-SIZED SYMBOLS  (ref: Section 4)
# ═══════════════════════════════════════════════════════════════
LARGE_OP_RULES = [
    _r(r'∑',  r'\\sum',        "∑"), _r(r'∏',  r'\\prod',       "∏"),
    _r(r'∐',  r'\\coprod',     "∐"), _r(r'∫',  r'\\int',        "∫"),
    _r(r'∮',  r'\\oint',       "∮"), _r(r'∬',  r'\\iint',       "∬"),
    _r(r'∭',  r'\\iiint',      "∭"), _r(r'⋂',  r'\\bigcap',     "⋂"),
    _r(r'⋃',  r'\\bigcup',     "⋃"), _r(r'⊎',  r'\\biguplus',   "⊎"),
    _r(r'⋁',  r'\\bigvee',     "⋁"), _r(r'⋀',  r'\\bigwedge',   "⋀"),
    _r(r'⨆',  r'\\bigsqcup',   "⨆"),
]

# ═══════════════════════════════════════════════════════════════
# 10. MISCELLANEOUS  (ref: Section 8)
# ═══════════════════════════════════════════════════════════════
MISC_RULES = [
    _r(r'∞',  r'\\infty',          "∞"), _r(r'∇',  r'\\nabla',          "∇"),
    _r(r'∂',  r'\\partial',        "∂"), _r(r'∀',  r'\\forall',         "∀"),
    _r(r'∃',  r'\\exists',         "∃"), _r(r'∄',  r'\\nexists',        "∄"),
    _r(r'∅',  r'\\emptyset',       "∅"), _r(r'∠',  r'\\angle',          "∠"),
    _r(r'√',  r'\\sqrt',           "√"), _r(r'…',  r'\\ldots',          "…"),
    _r(r'⋯',  r'\\cdots',          "⋯"), _r(r'⋮',  r'\\vdots',          "⋮"),
    _r(r'⋱',  r'\\ddots',          "⋱"), _r(r'ℏ',  r'\\hbar',           "ℏ"),
    _r(r'ℓ',  r'\\ell',            "ℓ"), _r(r'℘',  r'\\wp',             "℘"),
    _r(r'ℜ',  r'\\Re',             "ℜ"), _r(r'ℑ',  r'\\Im',             "ℑ"),
    _r(r'♣',  r'\\clubsuit',       "♣"), _r(r'♦',  r'\\diamondsuit',    "♦"),
    _r(r'♥',  r'\\heartsuit',      "♥"), _r(r'♠',  r'\\spadesuit',      "♠"),
    _r(r'★',  r'\\bigstar',        "★"), _r(r'◇',  r'\\Diamond',        "◇"),
    _r(r'△',  r'\\triangle',       "△"), _r(r'▽',  r'\\triangledown',   "▽"),
    _r(r'°',  r'^{\\circ}',        "°"), _r(r'′',  r"'",                "′"),
    _r(r'″',  r"''",               "″"), _r(r'‴',  r"'''",              "‴"),
    _r(r'∡',  r'\\measuredangle',  "∡"), _r(r'∢',  r'\\sphericalangle', "∢"),
    _r(r'ð',  r'\\eth',            "ð"), _r(r'§',  r'\\S',              "§"),
    _r(r'¶',  r'\\P',              "¶"), _r(r'©',  r'\\copyright',      "©"),
    _r(r'£',  r'\\pounds',         "£"),
]

# ═══════════════════════════════════════════════════════════════
# 11. MATH MODE ACCENTS  (ref: Section 9)
# ═══════════════════════════════════════════════════════════════
ACCENT_RULES = [
    _r(r'\\hat\s*\{',    r'\\hat{',    "hat"),
    _r(r'\\vec\s*\{',    r'\\vec{',    "vec"),
    _r(r'\\bar\s*\{',    r'\\bar{',    "bar"),
    _r(r'\\tilde\s*\{',  r'\\tilde{',  "tilde"),
    _r(r'\\dot\s*\{',    r'\\dot{',    "dot"),
    _r(r'\\ddot\s*\{',   r'\\ddot{',   "ddot"),
    _r(r'\\acute\s*\{',  r'\\acute{',  "acute"),
    _r(r'\\grave\s*\{',  r'\\grave{',  "grave"),
    _r(r'\\breve\s*\{',  r'\\breve{',  "breve"),
    _r(r'\\check\s*\{',  r'\\check{',  "check"),
]

# ═══════════════════════════════════════════════════════════════
# 12. FONT / STYLE FIXES  (ref: Section 11)
# ═══════════════════════════════════════════════════════════════
FONT_RULES = [
    _r(r'\\textbf\{([^}]+)\}',           r'\\mathbf{\1}',        "textbf to mathbf"),
    _r(r'\\text\{([^}]+)\}',             r'\\mathrm{\1}',        "text to mathrm"),
    _r(r'\\mathbb\{([a-z])\}',           r'\\mathbf{\1}',        "lowercase mathbb to mathbf"),
    _r(r'\\hat\{\\textbf\{([^}]+)\}\}',  r'\\hat{\\mathbf{\1}}', "hat{textbf} to hat{mathbf}"),
    _r(r'\\mathbf\{([^}=]+)=\}',         r'\\mathbf{\1}=',       "= leaked into mathbf"),
    _r(r'\\operatorname\s*\{',           r'\\operatorname{',     "operatorname spacing"),
]

# ═══════════════════════════════════════════════════════════════
# 13. STANDARD FUNCTIONS  (ref: Section 5)
# ═══════════════════════════════════════════════════════════════
FUNCTION_RULES = [
    # Bare function names not preceded by backslash → add backslash
    _r(
        r'(?<!\\)\b(arccos|arcsin|arctan|arg|cos|cosh|cot|coth|csc'
        r'|deg|det|dim|exp|gcd|hom|inf|ker|lg|liminf|limsup|lim'
        r'|ln|log|max|min|Pr|sec|sin|sinh|sup|tan|tanh)\b(?=\s*[\(\{])',
        r'\\\1',
        "bare function name"
    ),
]

# ═══════════════════════════════════════════════════════════════
# 14. PHYSICS-SPECIFIC
# ═══════════════════════════════════════════════════════════════
PHYSICS_RULES = [
    _r(r'\bd\s+([A-Z])\b',  r'd\1',  "d S → dS, d V → dV"),
    _r(r'\bd\s+([a-z])\b',  r'd\1',  "d x → dx, d q → dq"),
    _r(r'(\d)\s*(kg|ms|nm|km|mol|rad|eV|MeV|GeV|Hz|kHz|MHz|Pa|bar)',
       r'\1\\,\\mathrm{\2}', "SI unit spacing"),
]

# ═══════════════════════════════════════════════════════════════
# 15. CHEMISTRY-SPECIFIC
# ═══════════════════════════════════════════════════════════════
CHEMISTRY_RULES = [
    _r(r'\\mathrm\{([A-Z][a-z]?)\}(\d)',  r'\\ce{\1\2}',   "element+number to ce"),
    _r(r'\\rightleftharpoons',             r'\\ce{<=>}',    "equilibrium arrow"),
    _r(r'\\xrightarrow\{([^}]*)\}',        r'\\ce{->[\1]}', "labeled reaction arrow"),
]

# ═══════════════════════════════════════════════════════════════
# 16. FINAL WHITESPACE  (always last)
# ═══════════════════════════════════════════════════════════════
FINAL_RULES = [
    _r(r'\s+', ' ', "collapse whitespace"),
]


# ─────────────────────────────────────────────────────────────────
# Master pipeline — ORDER MATTERS
# ─────────────────────────────────────────────────────────────────
ALL_RULES = (
    CLEANUP_RULES   +
    SPACING_RULES   +
    GREEK_RULES     +
    LARGE_OP_RULES  +
    ARROW_RULES     +
    BINARY_OP_RULES +
    RELATION_RULES  +
    MISC_RULES      +
    ACCENT_RULES    +
    CONSTRUCT_RULES +
    DELIMITER_RULES +
    FONT_RULES      +
    FUNCTION_RULES  +
    PHYSICS_RULES   +
    CHEMISTRY_RULES +
    FINAL_RULES
)


# ─────────────────────────────────────────────────────────────────
# Public API — only function you need to call
# ─────────────────────────────────────────────────────────────────

def clean_latex(expr: str) -> str:
    for pattern, replacement, _ in ALL_RULES:
        expr = pattern.sub(replacement, expr)
    return expr.strip()


def fix_mathbb(expr: str) -> str:
    """Convert \\mathbb{X} to \\mathbf{X} (post-processing fix for pix2text output)."""
    return re.sub(r'\\mathbb\{([a-zA-Z])\}', r'\\mathbf{\1}', expr)