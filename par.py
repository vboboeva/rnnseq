L=0
m=2
loss='CE' # choice btw MSE and CE
# init=None # choice btw Const, Lazy, Rich 
init=None # choice btw Const, Lazy, Rich 
# init='Lazy' # choice btw Const, Lazy, Rich 
delay=0
# delay=4
task='RNNClass' # choice btw Class and Pred
# transfer=None # choice btw relu and tanh
transfer='relu' # choice btw relu and tanh
train_noise=0.000
n_epochs=101
lr=0.001
bs=1 # choice btw 2^n, n=0 to 6
frac_train=1.
n_sims=100
n_hidden=160
n_latent=10
ablate=False
epoch=-1
clustering='transitions' # choose btw 'class', 'timestep', 'hierarchical', 'transitions'
index_data=0
index_sim=0
snap_freq=5
cue_size=4
transferlearn=False
folder='modelWeightsDelta' # '', 'RNN_manysims' # 'results_old/results_pred/'
whichset='train'
take_correct=False
whichlayer='HiddenAct'

common_rotation='hidden' # 'weights' or 'hidden'
align_weight_deltas=True

n_types=7
alpha=10

compute_uc=True
which_t=1 # choose btw 1 and L

datasplit=0
classcomb=0

# 0.97 0.25 0.43 0.36
