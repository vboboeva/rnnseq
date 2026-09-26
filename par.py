"""
Configuration module for the analysis scripts (lowrank.py, analysis_utils.py,
lowrank_utils.py).

Every attribute here must match the parameters used during training in rnn.py,
because the analysis locates a trained run by rebuilding its output folder name
from these values:

    data/Task{task}_N{n_hidden}_nlatent{n_latent}_kSteps{k_steps}[scheduled]_
        L{L}_m{m}_alpha{alpha}_nepochs{n_epochs}_ntypes{n_types}_fractrain{frac_train:.1f}_obj{loss}
        _init{init}_transfer{transfer}_cuesize{cue_size}_delay{delay}
        _datasplit{datasplit}_{sim_idx}max_rank{max_rank}

    Use run_folder_name() to build this path (must match rnn.py output_folder_name).

To analyse a different sequence length, change `L` to one of the values present
in parameters.txt (the total input length seen by the RNN is cue_size + L).
"""

# ---- which trained run to analyse -------------------------------------------
folder = "test_rollout"          # root directory where rnn.py saves its outputs
L = 10                    # suffix length -> total sequence length = cue_size + L
datasplit = 0            # data-split / seed id (matches split_id in parameters.txt)
classcomb = 0            # which class-combination to load (index into classes.pkl)

# ---- task / model settings (must match rnn.py main_kwargs) ------------------
task = "RNNClass"        # the paper's core task: classify the abstract pattern
n_hidden = 64
n_latent = 10            # unused by RNNClass, but part of the folder name
m = 2                    # number of unique letters per pattern (AAB/ABA -> 2)
alpha = 10               # alphabet size
n_epochs = 50
n_types = 7              # number of pattern classes (aab, aba, abb)
frac_train = 0.8
loss = "CE"              # training objective (== rnn.py `objective`)
init = None              # weight init (== rnn.py `init_weights`)
transfer = "relu"        # hidden-unit nonlinearity (== rnn.py `transfer_func`)
cue_size = 4             # length of the base abstract pattern
delay = 0
train_noise = 0.0
k_steps = None           # == rnn.py main_kwargs / sys.argv[2] (0 -> None)
k_steps_scheduling = False  # == rnn.py sys.argv[3]; True appends "scheduled" after kSteps
sim_idx = 0              # freeze/transfer loop index in rnn.py (0 for a plain training run)
max_rank = None          # == rnn.py main_kwargs max_rank


def run_folder_name(L=None, datasplit=None):
    """Rebuild the output folder name used by rnn.py for a trained run."""
    L = L if L is not None else globals()["L"]
    datasplit = datasplit if datasplit is not None else globals()["datasplit"]
    k_steps_tag = f"kSteps{k_steps}"
    if k_steps_scheduling:
        k_steps_tag += "scheduled"
    return (
        f"Task{task}_N{n_hidden}_nlatent{n_latent}_"
        f"{k_steps_tag}_"
        f"L{L}_m{m}_alpha{alpha}_nepochs{n_epochs}_"
        f"ntypes{n_types}_fractrain{frac_train:.1f}_obj{loss}_"
        f"init{init}_transfer{transfer}_"
        f"cuesize{cue_size}_delay{delay}_datasplit{datasplit}_{sim_idx}"
        f"max_rank{max_rank}"
    )

# ---- analysis options -------------------------------------------------------
whichset = "train"       # analyse 'train' or 'test' sequences
common_rotation = "weights"  # align activity using 'weights' or 'hidden' SVs
out_bias = True          # the output layer (h2o) has a bias term
compute_uc = False       # skip the (expensive) ultrametric-content computation
clustering = "class"     # ordering used by analysis_utils overlap plots
