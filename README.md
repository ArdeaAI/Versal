# ArdEVO

Playground for evolutionary algo ML testing

The purpose of this is to test different ENAS methods for topology itself.

There is a custom dataset I made that goes through MANY rungs across modalities. The idea is to single in on a
search algorithm that seems to be able to generalize over the different rungs of the dataset, growing the topology
in minimum complexity needed as we go up each difficulty rung until we find at least an ideal algorithm, potentially
modified by me to something novel, that can be used across every rung and produce a significant score.

You can see an example of how to get tasks from the dataset in `ardevo/main.py` and the dataset is from: `https://huggingface.co/datasets/Ardea/Icarus-dataset`

I want to explore what we can do with `https://github.com/RobertTLange/evosax`, though I think we might have to come up with how to use that for topologies themselves instead of weights/hyperparameters. I also want to look into the different things mentioned in `https://github.com/rtu715/NAS-Bench-360`

To start out with, keep this small. For phase 1, let's just get the modular structure set up with the filesystem and an algorithm from evosax working on the first rung, which is just XOR.

We want to use ClearML for this as well as much as we can make use of it. I already have my config set up for that.

We want to modularize each part of this. I am not saying to use THIS file structure exactly, just giving an example of how to split things up into modules:
```
.
├── datasets
│   └── NexusDatasetSubsample -> /Users/sinjhin/WS/A/subworkdir/Experiments/datasets/NexusDatasetSubsample
├── docs
│   └── index.md
├── jupyter
├── models
│   ├── checkpoints
│   ├── evolved_substrates
│   └── sgd_substrates
├── nexus
│   ├── ablations
│   │   ├── configs.py
│   │   └── runner.py
│   ├── analysis
│   │   ├── cluster_analysis.py
│   │   ├── cross_modal.py
│   │   ├── metrics.py
│   │   ├── monitoring.py
│   │   └── visualization.py
│   ├── baselines
│   │   └── gradient_trainer.py
│   ├── core
│   │   ├── concepts.py
│   │   ├── levels.py
│   │   ├── nexus_core.py
│   │   └── relations.py
│   ├── evolution
│   │   ├── evolver.py
│   │   ├── results.py
│   │   └── strategies.py
│   ├── model
│   │   ├── blocks
│   │   └── model.py
│   ├── trials
│   │   └── nexus_trial.py
│   ├── tui
│   │   ├── bg_text
│   │   ├── app.py
│   │   ├── art.py
│   │   ├── background.py
│   │   ├── components.py
│   │   ├── screens.py
│   │   └── theme.py
│   ├── types
│   │   └── nexus_types.py
│   ├── utils
│   │   ├── collater.py
│   │   ├── config.py
│   │   ├── dataloader.py
│   │   ├── logging.py
│   │   ├── pipelines.py
│   │   └── proctor.py
│   └── main.py
├── notes
│   ├── archived_results
│   │   ├── concept_membership_report.md
│   │   ├── nexus_trial_20260529_232522.json
│   │   ├── nexus_trial_20260530_001901.json
│   │   ├── nexus_trial_20260530_001942.json
│   │   ├── nexus_trial_20260530_003039.json
│   │   ├── nexus_trial_20260530_003138.json
│   │   ├── nexus_trial_20260530_004036.json
│   │   ├── nexus_trial_20260530_004052.json
│   │   ├── nexus_trial_20260530_004124.json
│   │   ├── nexus_trial_20260530_004221.json
│   │   ├── nexus_trial_20260530_035704.json
│   │   ├── nexus_trial_20260530_035800.json
│   │   ├── nexus_trial_20260530_050412.json
│   │   ├── nexus_trial_20260530_062629.json
│   │   ├── nexus_trial_20260530_063519.json
│   │   ├── nexus_trial_20260530_121607.json
│   │   └── temporal_hierarchy_report.md
│   ├── 20251211_GPT_Chat.md
│   ├── 20251212_first_instructions.md
│   ├── ardea-vision.md
│   ├── conversation_before_project.md
│   ├── Initial_DeepResearch01-Opus.md
│   ├── initial_paper_idea.md
│   ├── NEXUS Notes.md
│   ├── notes.md
│   ├── project_structure.md
│   ├── roadmap.md
│   └── synopsis.md
├── paper
│   ├── drafts
│   │   ├── 20251201-NEXUS_PAPER
│   │   ├── 20250817-paper.md
│   │   └── nexus-position.md
│   ├── literature
│   │   ├── Integrated information theory - Wikipedia.pdf
│   │   └── Questioning Representational Optimism in Deep Learning.pdf
│   ├── neurips2025_style
│   │   ├── neurips_2025.pdf
│   │   ├── neurips_2025.sty
│   │   └── neurips_2025.tex
│   ├── results
│   │   ├── comparisons
│   │   ├── figures
│   │   └── metrics
│   └── submission
├── results
│   ├── nexus_trial_20260530_125305.json
│   └── nexus_trial_20260530_135532.json
├── scripts
│   ├── blind_eval.py
│   ├── distill_into_llm.py
│   ├── evaluate.py
│   ├── generate_figures.py
│   └── train.py
├── tests
│   ├── unit
│   │   └── core
│   └── conftest.py
├── tools
│   ├── config.smoke.toml
│   ├── format.py
│   ├── lint.py
│   ├── lintfix.py
│   └── test.py
├── CLAUDE.md
├── config.toml
├── LICENSE.md
├── pyproject.toml
├── README.md
└── uv.lock
```
