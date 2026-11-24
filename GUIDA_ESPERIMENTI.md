# Guida agli esperimenti (gioco completo)

Questa guida spiega come lanciare i 6 passi del piano sul gioco completo (14 giocatori, 2 poliziotti corrotti) con Qwen sulle GPU del cluster. Ogni comando va lanciato dalla cartella principale del progetto.

## Cosa c'è nel pacchetto

| Cartella / file | A cosa serve |
| --- | --- |
| `cpbench/` | Motore del gioco. Rispetto alla v0.1 aggiunge le varianti dello studio (radio, audit, canali, leakage), i backend LLM (`backends.py`) e un agente LLM che conta token ed errori di formato. |
| `experiments/evaluate.py` | Gioca N partite per cella in parallelo e salva metriche, intervalli di confidenza al 95%, token e transcript. |
| `experiments/judge.py` | Giudice LLM per l'inganno nel testo libero (chat). |
| `experiments/analyze.py` | Tabelle dei risultati, test di H1 (pendenza del lie rate sul livello di informazione), curve di training. |
| `experiments/configs/` | Configurazioni dei passi 1, 2 e 6, valutazione degli adapter addestrati, giudice, prova a secco. |
| `training/grpo.py` | Trainer GRPO multi-turno con LoRA: passo 3 (corrotto contro avversari fissi), passo 4 (self-play con popolazione), passo 5 (ablazioni). |
| `training/configs/` | Configurazioni base dei passi 3 e 4 (le ablazioni del passo 5 le cambiano da riga di comando). |
| `slurm/` | Script SLURM (server vLLM, training, valutazione) e `plan.py`, che genera tutti i job del piano. |
| `tests/` | 19 test, inclusi due test del trainer su un modello minuscolo su CPU. |

## Installazione sul cluster

```bash
cd $HOME && unzip corrupt-police-bench-experiments.zip && cd corrupt-police-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt
python -m pytest -q tests                      # deve dare 19 passed
```

1. Apri `slurm/env.sh` e imposta partizione, account, percorso del progetto e cache dei modelli.
2. Scarica i pesi una volta sola da un nodo di login: `huggingface-cli download Qwen/Qwen3.5-9B` (e `Qwen/Qwen3.5-4B`, `Qwen/Qwen3-14B` per l'ablazione sulla taglia). Controlla che gli id esatti esistano su Hugging Face: li ho scritti a mano.
3. Prova tutto senza GPU e senza modello: `python experiments/evaluate.py --config experiments/configs/dryrun_mock.yaml`. Gioca 16 partite con un "modello finto" che risponde mosse legali a caso.

## I 6 passi

`python slurm/plan.py` scrive in `jobs/` uno script di sottomissione per passo (`--minimal` per la versione minima del documento dei costi). Si lancia un passo alla volta, guardando i risultati prima del successivo.

### Passo 1: sanity check zero-shot

```bash
bash jobs/submit_step1.sh          # = sbatch slurm/eval.sbatch experiments/configs/step1_sanity.yaml
```

Qwen non addestrato gioca i 2 corrotti contro agenti euristici (50 partite), poi 5 partite con Qwen in tutti i 14 seggi. Guarda in `runs/step1_sanity/*/aggregate.json`:

- `llm_parse_failure_rate.corrupt`: risposte non leggibili come JSON. **Se supera il 5–10%**, usa il 14B o correggi il prompt prima di andare avanti.
- `usage.json`: token per partita e secondi per partita, per ricalcolare le ore GPU dei passi successivi.
- I transcript in `transcripts/` si aprono con il viewer (`viz/`), così vedi come parla il modello.
- Con 14 giocatori la storia di 80 eventi copre circa un round. Se il modello sembra "dimenticare", alza `history_events` (costa token).

### Passo 2: baseline

```bash
bash jobs/submit_step2.sh
```

Gioca 200 partite per cella nelle 4 condizioni del passo 3 (radio esatta o assente × audit 0 o 0,3), con corrotto casuale, euristico e Qwen non addestrato. Questi sono i punti di partenza con cui confrontare il training. Usano gli stessi seed di valutazione (da 10000) di tutti i passi successivi.

### Passo 3: RL del corrotto contro avversari fissi

```bash
bash jobs/submit_step3.sh          # 12 run (4 condizioni × 3 seed), 6 con --minimal
```

Ogni run addestra un adapter LoRA condiviso dai 2 corrotti, con reward uguale al payout finale e polizia e ladri euristici. Il job si rimette in coda da solo prima del limite di tempo e riprende dall'ultimo step. Alla fine parte automaticamente la valutazione su 200 partite (`runs/eval/<nome>`).

Mentre gira:

```bash
tail -f runs/train/<nome>/train_log.jsonl
python experiments/analyze.py curves runs/train/step3_*      # reward, lie rate, detection, errori di formato
```

Criterio di successo del MVP: `payout_mean.corrupt` del modello addestrato maggiore di quello del corrotto euristico (passo 2), con intervalli che non si sovrappongono. Per H1: `python experiments/analyze.py h1 runs/eval/step3_*`.

### Passo 4: self-play MARL

```bash
bash jobs/submit_step4.sh          # parte dai corrotti del passo 3 (seed corrispondente)
```

Corrotti e poliziotti onesti sono due adapter sullo stesso modello e si addestrano a turni. L'avversario è l'ultima versione dell'altro ruolo con probabilità 0,5, altrimenti uno snapshot passato (ogni 10 step). Per H3 confronta nel log, step per step, la reward e il lie rate dei corrotti con la `corrupt_detection_rate`.

### Passo 5: ablazioni

```bash
bash jobs/submit_step5.sh          # 33 run (11 varianti × 3 seed), 11 con --minimal
```

| Variante | Cosa cambia |
| --- | --- |
| `size_4b`, `size_14b` | Modello base: Qwen3.5 4B, Qwen3 14B (la famiglia Qwen3.5 non ha un 14B). |
| `comm_actions_only`, `comm_reports`, `comm_chat_no_dm` | Canale: nessuna comunicazione; solo report; chat senza messaggi privati. La condizione completa è quella del passo 3. |
| `info_radio_delayed`, `info_radio_zone`, `info_radio_and_pieces` | Informazione del corrotto: radio in ritardo di un round; solo una zona; radio più mappa dei pezzi. Esatta e assente sono già nel passo 3. |
| `leak_0.1`, `leak_0.3` | Probabilità che un pensiero privato venga mostrato ai compagni di squadra (H5). |
| `reward_deception` | Condizione di controllo: payout più un premio esplicito per aver allontanato i compagni dalla verità. |

### Passo 6: valutazione finale con i modelli frontier

```bash
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
sbatch slurm/eval_step6.sbatch runs/train/step3_radio_exact_audit0_s1 --episodes 20    # prima una prova
sbatch slurm/eval_step6.sbatch runs/train/step3_radio_exact_audit0_s1                  # poi 200 partite
python experiments/judge.py --runs runs/step6_frontier --config experiments/configs/judge.yaml
python experiments/analyze.py table runs/step2_baselines runs/eval runs/step6_frontier > risultati.md
```

Le celle `A_*` mettono un modello frontier nei seggi corrotti contro la polizia euristica. Le celle `B_*` mettono Sonnet come polizia contro il Qwen addestrato. **Questo è l'unico passo che costa soldi:** con 20 partite misuri il costo reale (`usage.json`) prima di lanciarne 200. Prima di usare il giudice su tutto, confrontalo con circa 200 messaggi etichettati a mano.

## Varianti del gioco (nuovi parametri di `GameConfig`)

| Parametro | Valori | Effetto |
| --- | --- | --- |
| `radio_mode` | `exact`, `delayed`, `zone`, `none` | Informazione privata dei corrotti sulla posizione dei ladri. In `zone` il ladro sta sempre dentro la zona comunicata. |
| `radio_zone_radius` | intero (default 2) | Raggio della zona. |
| `audit_prob` | 0–1 | Probabilità che un report di squadra venga verificato e la verità mostrata alla squadra. |
| `comm_mode` | `actions_only`, `reports`, `chat`, `full` | Canali disponibili. |
| `thought_leak_prob` | 0–1 | Probabilità che un "thought" privato arrivi ai compagni. |
| `corrupt_know_pieces` | true/false | (già esistente) il corrotto conosce anche i pezzi. |

Audit e leakage usano un generatore casuale separato, quindi con lo stesso seed ruoli, pezzi e spareggi restano identici tra condizioni. Le regole nel prompt sono generate dalla configurazione, quindi descrivono sempre la variante giocata.

## Cosa ho testato e cosa no

- **Testato su CPU:** motore e varianti, runner di valutazione, giudice, analisi, e il trainer GRPO completo (rollout, advantage di gruppo, update, snapshot, ripresa) su un modello minuscolo generato al volo, con la generazione via transformers.
- **Non testato (manca la GPU qui):** generazione con vLLM (in-process e server), Qwen3.5 9B reale, QLoRA, gli script SLURM sul vostro cluster. La prima esecuzione del passo 1 e un run del passo 3 con `--set train.steps=2` servono a verificarli.
- **Da controllare al primo uso:** che `chat_template_kwargs: {enable_thinking: false}` spenga davvero il ragionamento di Qwen3.5 (altrimenti le risposte sono lunghissime) e che `target_modules: all-linear` trovi i layer giusti dell'architettura Qwen3.5.
- **Non incluso:** PPO con value head (c'è solo GRPO, che è quello previsto dal paper per il tier C) e il tier B (LLM congelato + selettore RL), che non è nei 6 passi.

## Tempi di un run sul gioco completo

Un passo di training gioca 32 partite (4 seed × 8) e addestra al massimo 2.048 decisioni. Sul gioco completo ogni corrotto prende circa 88 decisioni a partita, quindi un run da 300 step costa circa 4–5 volte un run sul mini del documento dei costi. Le leve per accorciarlo sono `train.steps`, `train.group_size`, `train.max_samples_per_step` e `history_events`. Misura i secondi per step (`t_total_s` nel log) nei primi step e ricalibra.
