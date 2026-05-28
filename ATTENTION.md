# Attention Mechanism — Bilkul Scratch Se, Numbers Ke Saath

> Is file mein woh sab hai jo embedding se lekar attention output tak hota hai.
> Koi assumptions nahi — har step concrete numbers ke saath.

---

## Part 1 — Word Ko Number Kaise Milta Hai

### Step 1: Tokenizer

```
"hello world"
     ↓
Tokenizer
     ↓
"hello" = 9906
"world" = 1917
```

Har word ka ek fixed index number hota hai vocabulary mein.
Vocabulary mein 100,277 words hain (cl100k_base).

---

### Step 2: Embedding Table — Har Word Ka "Ghar"

```
Embedding Table:  [100,277 rows  ×  4 columns]   (simplified, real = 128 columns)

Row       Word         Numbers (embedding vector)
────────────────────────────────────────────────
Row 0     <eos>        [ 0.0,  0.0,  0.0,  0.0 ]
...
Row 9906  hello        [ 1,    2,    3,    4   ]   ← "hello" ka ghar
Row 9907  Hello        [ 1.1,  2.1,  3.0,  3.9 ]
...
Row 1917  world        [ 4,    3,    2,    1   ]   ← "world" ka ghar
...
Row 100276 (last)      [ 0.0,  0.0,  0.0,  0.0 ]
```

**Kya hai yeh numbers?**
```
Training shuru hone se pehle: sab random numbers (koi matlab nahi)
Training ke baad:             numbers meaningful ho jaate hain

  Similar words → similar vectors:
    "hello" = [1,  2,  3,  4]
    "hi"    = [0.9, 2.1, 2.8, 4.2]   ← kaafi similar
    "car"   = [8,  -3,  1,  -5]      ← bilkul alag
```

---

## Part 2 — Weight Matrix Kya Hai

### W kahan se aaya?

```
W (Weight Matrix) — kisi word ka embedding NAHI hai

Training shuru hone se pehle:
  W = random numbers, koi matlab nahi

Training ke baad:
  W = "patterns ko kaise transform karo" seek liya

Size: 4×4  (agar d_model=4 ho)
      128×128 (real model mein)
```

### Embedding vs Weight Matrix — Fark

```
                 Embedding Table          Weight Matrix W
                 ───────────────          ───────────────
Kya hai?         Word lookup table        Transformation recipe
Kahan se?        Har word ka alag row     Kisi word se nahi
Size             100,277 × 4              4 × 4
Kaam             Word ki identity store   Words ko transform karo
Training mein    Word meanings seekhe     Patterns seekhe
```

---

## Part 3 — v × W Kya Karta Hai (Concrete Math)

### Setup

```
"hello" ka embedding:
  v = [1, 2, 3, 4]

Weight matrix W (4×4):

        col0  col1  col2  col3
row0  [  1,    1,    0,    0  ]
row1  [  1,    0,    1,    0  ]
row2  [  0,    1,    0,    1  ]
row3  [  0,    0,    1,    1  ]
```

### v × W — Har Step

```
Naya number 0 = v[0]×W[0][0] + v[1]×W[1][0] + v[2]×W[2][0] + v[3]×W[3][0]
              =  1  ×   1    +  2  ×   1    +  3  ×   0    +  4  ×   0
              =     1        +     2        +     0        +     0
              = 3

Naya number 1 = v[0]×W[0][1] + v[1]×W[1][1] + v[2]×W[2][1] + v[3]×W[3][1]
              =  1  ×   1    +  2  ×   0    +  3  ×   1    +  4  ×   0
              =     1        +     0        +     3        +     0
              = 4

Naya number 2 = v[0]×W[0][2] + v[1]×W[1][2] + v[2]×W[2][2] + v[3]×W[3][2]
              =  1  ×   0    +  2  ×   1    +  3  ×   0    +  4  ×   1
              =     0        +     2        +     0        +     4
              = 6

Naya number 3 = v[0]×W[0][3] + v[1]×W[1][3] + v[2]×W[2][3] + v[3]×W[3][3]
              =  1  ×   0    +  2  ×   0    +  3  ×   1    +  4  ×   1
              =     0        +     0        +     3        +     4
              = 7

"hello"  →  W  →  [3, 4, 6, 7]
```

### Same W, Alag Word

```
"world" ka embedding: v = [4, 3, 2, 1]
Same W matrix

Naya number 0 = 4×1 + 3×1 + 2×0 + 1×0 = 7
Naya number 1 = 4×1 + 3×0 + 2×1 + 1×0 = 6
Naya number 2 = 4×0 + 3×1 + 2×0 + 1×1 = 4
Naya number 3 = 4×0 + 3×0 + 2×1 + 1×1 = 3

"world"  →  W  →  [7, 6, 4, 3]
```

### Kya Hua?

```
          Embedding          Same W          Output
          ─────────────────────────────────────────
"hello"   [1, 2, 3, 4]  →   W    →   [3, 4, 6, 7]
"world"   [4, 3, 2, 1]  →   W    →   [7, 6, 4, 3]

Ek hi W — alag input — alag output ✅
```

**W ne kya kiya?**
```
W ki row0 = [1, 1, 0, 0]  → "input 0 aur input 1 mila do"
W ki row2 = [0, 1, 0, 1]  → "input 1 aur input 3 mila do"

W = mixing recipe
Training mein W seekhta hai: kaunse features kaise milao
```

---

## Part 4 — Teen W Matrices: Q, K, V

Attention layer mein sirf ek W nahi hoti — TEEN hoti hain:

```
v (embedding)  →  × W_q  →  Q   (Query)
v (embedding)  →  × W_k  →  K   (Key)
v (embedding)  →  × W_v  →  V   (Value)
```

### Har W ka Alag Kaam

```
W_q se Q nikalti hai:
  "Main kya dhundh raha hoon?"
  "Mujhe kisi se kya information chahiye?"

W_k se K nikalti hai:
  "Mere paas kya information hai?"
  "Main kaunsi context offer kar sakta hoon?"

W_v se V nikalti hai:
  "Meri actual information kya hai?"
  "Agar koi mujhe choose kare, toh main kya dunga?"
```

### Example — Teen Alag W Matrices

```
W_q (Query matrix):         W_k (Key matrix):          W_v (Value matrix):
[ 1, 1, 0, 0 ]             [ 0, 1, 1, 0 ]             [ 1, 0, 0, 0 ]
[ 1, 0, 1, 0 ]             [ 1, 0, 0, 1 ]             [ 0, 1, 0, 0 ]
[ 0, 1, 0, 1 ]             [ 0, 1, 0, 1 ]             [ 0, 0, 1, 0 ]
[ 0, 0, 1, 1 ]             [ 1, 0, 1, 0 ]             [ 0, 0, 0, 1 ]
```

```
"hello" [1,2,3,4] se:
  Q_hello = [1,2,3,4] × W_q = [3, 4, 6, 7]   ← hello kya dhundh raha hai

"world" [4,3,2,1] se:
  K_world = [4,3,2,1] × W_k = [5, 6, 5, 6]   ← world kya offer karta hai
  V_world = [4,3,2,1] × W_v = [4, 3, 2, 1]   ← world ki actual information
```

---

## Part 5 — Attention Score: Q · K

Q aur K ka dot product = score = "kitna dhyan dena chahiye"

```
Q_hello = [3, 4, 6, 7]    (hello kya dhundh raha hai)
K_world = [5, 6, 5, 6]    (world kya offer karta hai)

Score = Q · K
      = 3×5  +  4×6  +  6×5  +  7×6
      = 15   +  24   +  30   +  42
      = 111

High score (111) = "hello" aur "world" bahut zyada match karte hain
                 = "hello" should pay a lot of attention to "world"
```

### Multiple Words Ka Example

```
Sentence: "hello world cat"

  Embeddings:
    hello = [1, 2, 3, 4]
    world = [4, 3, 2, 1]
    cat   = [2, 2, 2, 2]

  Q_hello = [3, 4, 6, 7]

  K_hello = [3, 4, 4, 4]   (hello ka apna key)
  K_world = [5, 6, 5, 6]   (world ka key)
  K_cat   = [4, 4, 4, 4]   (cat ka key)

  Scores for "hello":
    hello → hello = [3×3 + 4×4 + 6×4 + 7×4] = 9+16+24+28 = 77
    hello → world = [3×5 + 4×6 + 6×5 + 7×6] = 15+24+30+42 = 111  ← highest
    hello → cat   = [3×4 + 4×4 + 6×4 + 7×4] = 12+16+24+28 = 80

  "hello" sabse zyada "world" pe dhyan dega (score 111)
  "cat" se thoda kam (80)
  Khud se sabse kam (77)
```

---

## Part 6 — Softmax: Scores Ko Weights Mein Badlo

```
Raw scores:   hello=77,  world=111,  cat=80

Softmax (0 se 1 ke beech, sab milake = 1.0):
  hello weight = 0.15   (15% dhyan)
  world weight = 0.72   (72% dhyan)  ← sabse zyada
  cat   weight = 0.13   (13% dhyan)
```

---

## Part 7 — Value Se Final Output

```
V_hello = [1, 2, 3, 4]   (hello ki value)
V_world = [4, 3, 2, 1]   (world ki value)
V_cat   = [2, 2, 2, 2]   (cat ki value)

Attention weights:
  hello = 0.15
  world = 0.72
  cat   = 0.13

"hello" ka final output:
  = 0.15 × V_hello  +  0.72 × V_world  +  0.13 × V_cat
  = 0.15 × [1,2,3,4]  +  0.72 × [4,3,2,1]  +  0.13 × [2,2,2,2]

Number 0: 0.15×1 + 0.72×4 + 0.13×2 = 0.15 + 2.88 + 0.26 = 3.29
Number 1: 0.15×2 + 0.72×3 + 0.13×2 = 0.30 + 2.16 + 0.26 = 2.72
Number 2: 0.15×3 + 0.72×2 + 0.13×2 = 0.45 + 1.44 + 0.26 = 2.15
Number 3: 0.15×4 + 0.72×1 + 0.13×2 = 0.60 + 0.72 + 0.26 = 1.58

"hello" ka attention output = [3.29, 2.72, 2.15, 1.58]
```

---

## Part 8 — Poora Flow Ek Jagah

```
Input: "hello world cat"

Step 1 — Tokenize:
  hello → 9906
  world → 1917
  cat   → 2564

Step 2 — Embedding lookup:
  9906 → [1, 2, 3, 4]
  1917 → [4, 3, 2, 1]
  2564 → [2, 2, 2, 2]

Step 3 — Q, K, V nikalo (har word ke liye):
  hello: Q=[3,4,6,7]   K=[3,4,4,4]   V=[1,2,3,4]
  world: Q=[7,6,4,3]   K=[5,6,5,6]   V=[4,3,2,1]
  cat:   Q=[4,4,6,6]   K=[4,4,4,4]   V=[2,2,2,2]

Step 4 — Attention scores (Q · K):
  "hello" ke scores:
    → hello: 77
    → world: 111  ← max
    → cat:   80

Step 5 — Softmax → weights:
  hello=0.15, world=0.72, cat=0.13

Step 6 — Weighted sum of Values:
  "hello" output = 0.15×V_hello + 0.72×V_world + 0.13×V_cat
                 = [3.29, 2.72, 2.15, 1.58]

Step 7 — O projection (W_o se multiply):
  [3.29, 2.72, 2.15, 1.58] × W_o → final output vector

Step 8 — FFN se guzro (alag transformation)

Step 9 — × N layers repeat karo

Step 10 — lm_head → 100,277 probabilities → next word
```

---

## Part 9 — "hello" Pehle vs Baad

```
Pehle (sirf embedding):
  "hello" = [1, 2, 3, 4]
  Sirf word ki identity — context ka koi idea nahi

Baad (attention ke baad):
  "hello" = [3.29, 2.72, 2.15, 1.58]
  Word ki identity + "world" aur "cat" ka context include hai

Yahi hai attention ka magic:
  Har word sirf khud nahi raha
  Aas paas ke words ka context absorb kar leta hai
```

---

## Part 10 — Training Mein Kya Seekha Jaata Hai

```
Embedding table (12.8M numbers):
  Seekhta hai: har word ka best 4-dimensional (ya 128-dim) representation

W_q (16,384 numbers):
  Seekhta hai: "query banane ka best tarika"

W_k (16,384 numbers):
  Seekhta hai: "key banane ka best tarika"

W_v (16,384 numbers):
  Seekhta hai: "value banane ka best tarika"

W_o (16,384 numbers):
  Seekhta hai: "final output combine karne ka best tarika"

Sab milake ek language model banta hai:
  Input words → Q,K,V → attention scores → context-aware output → next word
```

---

## Summary Table

```
Component    Kya hai?                      Size (d=4)    Kahan se?
────────────────────────────────────────────────────────────────────
Embedding    Har word ka ghar              100277 × 4    Word-specific
W_q          Query banana ki recipe        4 × 4         Shared, trained
W_k          Key banana ki recipe          4 × 4         Shared, trained
W_v          Value banana ki recipe        4 × 4         Shared, trained
W_o          Output combine karna          4 × 4         Shared, trained
Q            "main kya dhundh raha hoon"   1 × 4         v × W_q
K            "mere paas kya hai"           1 × 4         v × W_k
V            "meri actual info"            1 × 4         v × W_v
Score        Q · K = kitna dhyan dena      1 number      dot product
Weight       Softmax(score)                0.0 to 1.0    normalized
Output       Context-aware representation  1 × 4         weighted sum of V
```
