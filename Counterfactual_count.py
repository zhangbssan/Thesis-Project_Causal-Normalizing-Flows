import glob

import wandb
import os
import causal_nf.config as causal_nf_config
import causal_nf.utils.training as causal_nf_train
from causal_nf.config import cfg
import torch
from torch import Tensor
from causal_nf.preparators.scm import SCMPreparator
from causal_nf import modules
from causal_nf import sem_equations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make a config file
args_list, args = causal_nf_config.parse_args()

config = causal_nf_config.build_config(
    config_file=args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)
cfg.seed=2

# prepare data
preparator_data = SCMPreparator.loader(cfg.dataset)
preparator_data.prepare_data()
loader = preparator_data._data_loader(
        preparator_data.datasets[0], batch_size=len(preparator_data.datasets[0]), shuffle=False, num_workers=0
    )
x, y = next((iter(loader)))
scaler = preparator_data.get_scaler()
interven_value = torch.tensor([2.5,4,5,4])
x_mean = torch.mean(x,axis=0)
x_std  = torch.std (x,axis=0)

def ckpt_file_name(ckpt_name_list):
   for i in range(len(ckpt_name_list)):
       if 'epoch' in ckpt_name_list[i]:
           name =ckpt_name_list[i]
   return name


## Step 3.1: Framework with adjacency matrix
def preparator_framework(cfg):
   ## framework with some adjacency matrix
   preparator_model =SCMPreparator.loader(cfg)
   preparator_model.prepare_data()
   return preparator_model


## Step 3.2: Load parameters(.ckpt file)
def load_ckpt(model_name):
  dirpath = os.path.join(cfg.root_dir, model_name)
  ckpt_name_list = glob.glob(os.path.join(dirpath, f"*ckpt"))
  ckpt_name = ckpt_file_name(ckpt_name_list)
  return ckpt_name

## Step 3.3: Configuration models
def models(cfg,dataset, model_name):
  ## framework
  preparator_model = preparator_framework(dataset)
  ## two models
     ## model with real causal graph
  model = causal_nf_train.load_model(
            cfg=cfg, preparator=preparator_model, ckpt_file= load_ckpt(model_name)
        )
  model = model.model
  model.eval()
  return model


## Step 4: Count counterfactual error for one feature
def count_couterfactual_error(model_real,model2,index:int):

   #my_index = torch.tensor(range(0,len(x[0])))
   #print("---")
   output_cf = {}
   output_cf_wrong_1 ={}
   output_cf_error_mean ={}
   output_cf_error_std ={}

   output_cf_index= model_real.compute_counterfactual(
                    x, index, value=interven_value[index],scaler = scaler

    ) ## counterfactual for the real graph
   output_cf_wrong_index = model2.compute_counterfactual(
        x, index, value=interven_value[index],scaler = scaler
    ) ## counterfactual for one wrong graph
   output_cf[f"{index}"]= output_cf_index
   output_cf_wrong_1[f"{index}"]= output_cf_wrong_index
   output_cf_error_mean[f"{index}"] = torch.mean(torch.sqrt(torch.sub(output_cf_index,output_cf_wrong_index)**2),axis=0)
   output_cf_error_std[f"{index}"] = torch.std(torch.sqrt(torch.sub(output_cf_index,output_cf_wrong_index)**2),axis=0)
   return output_cf_error_mean,output_cf_error_std




## prepare the model
#scaler = preparator_data.get_scaler()

## Get the ground truth counterfactual
def get_ground_truth_SCM(name:str):
    if name in config["dataset__name"]:
        functions = sem_equations.Simpson(config["dataset__sem_name"]).functions
        inverses = sem_equations.Simpson(config["dataset__sem_name"]).inverses
        # s = torch.nn.functional.softplus  
    return functions,inverses

def update_U(X_factual:Tensor, inverses)-> Tensor:
    u_obs = torch.zeros(len(X_factual), 4)  ## modify the number
    for i in range(0, len(X_factual)):
        # get the u
        u_obs[i] = torch.tensor([inverses[0](X_factual[i][0]),
                                 inverses[1](X_factual[i][0], X_factual[i][1]),
                                 inverses[2](X_factual[i][0], X_factual[i][1], X_factual[i][2]),
                                 inverses[3](X_factual[i][0], X_factual[i][1], X_factual[i][2], X_factual[i][3])])
    return u_obs
def ground_counterfactual(X: Tensor, u_obs:Tensor, index: int,interven_value:Tensor,functions,inverses) -> Tensor:
    x_factual=X.clone()
    # u_obs = torch.zeros(len(X_factual),4) ## modify the number
    U_obs=u_obs.clone()
    u_temp = torch.zeros(len(x_factual), 4)  ## modify the number
    x_cf = torch.zeros(len(x_factual), 4)  ## modify the number
    x_factual[:,index]=interven_value
    for i in range(0, len(x_factual)):
        if index ==0:
            x_cf[i]=torch.tensor([x_factual[i][0],
                                  functions[1](x_factual[i][0],
                                               U_obs[i][1]),
                                  functions[2](x_factual[i][0],
                                               functions[1](x_factual[i][0],
                                               U_obs[i][1]),
                                               U_obs[i][2]),
                                  functions[3](x_factual[i][0],
                                               functions[1](x_factual[i][0], U_obs[i][1]),
                                               functions[2](x_factual[i][0], functions[1](x_factual[i][0], U_obs[i][1]), U_obs[i][2]),
                                               U_obs[i][3])])
        if index ==1:
            x_cf[i] = torch.tensor([functions[0](U_obs[i][0]),
                                    x_factual[i][1],
                                    functions[2](functions[0](U_obs[i][0]),
                                                 x_factual[i][1],
                                                 U_obs[i][2]),
                                    functions[3](functions[0](U_obs[i][0]),
                                                 x_factual[i][1],
                                                 functions[2](functions[0](U_obs[i][0]),x_factual[i][1], U_obs[i][2]),
                                                 U_obs[i][3])])
        if index ==2:
            x_cf[i] = torch.tensor([functions[0](x_factual[i][0]),
                                    functions[1](functions[0](x_factual[i][0]), U_obs[i][1]),
                                    x_factual[i][2],
                                    functions[3](functions[0](x_factual[i][0]),functions[1](functions[0](x_factual[i][0]), U_obs[i][1]),x_factual[i][2], U_obs[i][3])])
        if index ==3:
            x_cf[i] = torch.tensor([functions[0](x_factual[i][0]),
                                    functions[1](functions[0](x_factual[i][0]),
                                                 U_obs[i][1]),
                                    functions[2](functions[0](x_factual[i][0]),
                                                 functions[1](functions[0](x_factual[i][0]), U_obs[i][1]),
                                                 U_obs[i][2]),
                                    x_factual[i][3]])
    return x_cf


def dict_to_latex_table(mean,std,mean_wrong_graph,std_wrong_graph,mean_wrong_graph_real_graph,std_wrong_graph_real_graph):
    # Extracting column names
    columns = ['{}','{}','$do(x_1=2.5)$','$do(x_2=4)$','$do(x_3=5)$','$do(x_4=4)$'] # modify every time
    # Extracting rows from the dictionary
    means = list(zip(*mean.values()))
    stds  = list(zip(*std.values()))

    means_wg= list(zip(*mean_wrong_graph.values()))
    stds_wg= list(zip(*std_wrong_graph.values()))

    means_wg_rg=list(zip(*mean_wrong_graph_real_graph.values()))
    stds_wg_rg= list(zip(*std_wrong_graph_real_graph.values()))
    
    # Starting the LaTeX table
    latex_code = "\\begin{table}[]\n"
    latex_code +="\\centering\n"
    latex_code += "\\begin{tabular}{p{1.3cm} p{0.3cm} p{2.3cm} p{2.3cm} p{2.3cm} p{2.3cm}}\n"
    latex_code += "\\toprule\n"
    latex_code +="\multicolumn{6}{c}{\\textbf{The Distribution of Counterfactual Errors(Mean$\pm $ Std)}}"+" \\\\\n"
    latex_code += "\\midrule\n"
    latex_code += " & ".join(columns) + " \\\\\n"
    latex_code += "\\midrule\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_1 $}"+" & "+ str(round(means[0][0].item(),4)) + "$\pm$"+str(round(stds[0][0].item(),4))+" & "+str(round(means[0][1].item(),4)) + "$\pm$"+str(round(stds[0][1].item(),4))+  " & "+str(round(means[0][2].item(),4)) + "$\pm$"+str(round(stds[0][2].item(),4))+ " & "+str(round(means[0][3].item(),4)) + "$\pm$"+str(round(stds[0][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_2 $}"+" & "+ str(round(means[1][0].item(),4)) + "$\pm$"+str(round(stds[1][0].item(),4))+ " & "+str(round(means[1][1].item(),4)) + "$\pm$"+str(round(stds[1][1].item(),4))+ " & "+str(round(means[1][2].item(),4)) + "$\pm$"+str(round(stds[1][2].item(),4))+" & "+str(round(means[1][3].item(),4)) + "$\pm$"+str(round(stds[1][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{$a_{ij}=0$}"+"&"+"\multicolumn{1}{c}{$x^{cf}_3 $}"+" & "+str(round(means[2][0].item(),4)) + "$\pm$"+str(round(stds[2][0].item(),4))+ " & "+str(round(means[2][1].item(),4)) + "$\pm$"+str(round(stds[2][1].item(),4))+  " & "+str(round(means[2][2].item(),4)) + "$\pm$"+str(round(stds[2][2].item(),4))+ " & "+str(round(means[2][3].item(),4)) + "$\pm$"+str(round(stds[2][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_4 $}"+" & "+ str(round(means[3][0].item(),4)) + "$\pm$"+str(round(stds[3][0].item(),4))+ " & "+str(round(means[3][1].item(),4)) + "$\pm$"+str(round(stds[3][1].item(),4))+ " & "+str(round(means[3][2].item(),4)) + "$\pm$"+str(round(stds[3][2].item(),4))+" & "+str(round(means[3][3].item(),4)) + "$\pm$"+str(round(stds[3][3].item(),4))+" \\\\\n"
    latex_code += "\\midrule\n"

    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_1 $}"+" & "+ str(round(means_wg[0][0].item(),4)) + "$\pm$"+str(round(stds_wg[0][0].item(),4))+" & "+str(round(means_wg[0][1].item(),4)) + "$\pm$"+str(round(stds_wg[0][1].item(),4))+  " & "+str(round(means_wg[0][2].item(),4)) + "$\pm$"+str(round(stds_wg[0][2].item(),4))+ " & "+str(round(means_wg[0][3].item(),4)) + "$\pm$"+str(round(stds_wg[0][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_2 $}"+" & "+ str(round(means_wg[1][0].item(),4)) + "$\pm$"+str(round(stds_wg[1][0].item(),4))+ " & "+str(round(means_wg[1][1].item(),4)) + "$\pm$"+str(round(stds_wg[1][1].item(),4))+ " & "+str(round(means_wg[1][2].item(),4)) + "$\pm$"+str(round(stds_wg[1][2].item(),4))+ " & "+str(round(means_wg[1][3].item(),4)) + "$\pm$"+str(round(stds_wg[1][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_3 $}"+" & "+ str(round(means_wg[2][0].item(),4)) + "$\pm$"+str(round(stds_wg[2][0].item(),4))+ " & "+str(round(means_wg[2][1].item(),4)) + "$\pm$"+str(round(stds_wg[2][1].item(),4))+  " & "+str(round(means_wg[2][2].item(),4)) + "$\pm$"+str(round(stds_wg[2][2].item(),4))+  " & "+str(round(means_wg[2][3].item(),4)) + "$\pm$"+str(round(stds_wg[2][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_4 $}"+" & "+ str(round(means_wg[3][0].item(),4)) + "$\pm$"+str(round(stds_wg[3][0].item(),4))+  " & "+str(round(means_wg[3][1].item(),4)) + "$\pm$"+str(round(stds_wg[3][1].item(),4))+ " & "+str(round(means_wg[3][2].item(),4)) + "$\pm$"+str(round(stds_wg[3][2].item(),4))+" & "+str(round(means_wg[3][3].item(),4)) + "$\pm$"+str(round(stds_wg[3][3].item(),4))+" \\\\\n"
    latex_code += "\\midrule\n"

    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_1 $}"+" & "+ str(round(means_wg_rg[0][0].item(),4)) + "$\pm$"+str(round(stds_wg_rg[0][0].item(),4))+ " & "+str(round(means_wg_rg[0][1].item(),4)) + "$\pm$"+str(round(stds_wg_rg[0][1].item(),4))+ " & "+str(round(means_wg_rg[0][2].item(),4)) + "$\pm$"+str(round(stds_wg_rg[0][2].item(),4))+ " & "+str(round(means_wg_rg[0][3].item(),4)) + "$\pm$"+str(round(stds_wg_rg[0][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_2 $}"+" & "+ str(round(means_wg_rg[1][0].item(),4)) + "$\pm$"+str(round(stds_wg_rg[1][0].item(),4))+  " & "+str(round(means_wg_rg[1][1].item(),4)) + "$\pm$"+str(round(stds_wg_rg[1][1].item(),4))+ " & "+str(round(means_wg_rg[1][2].item(),4)) + "$\pm$"+str(round(stds_wg_rg[1][2].item(),4))+" & "+str(round(means_wg_rg[1][3].item(),4)) + "$\pm$"+str(round(stds_wg_rg[1][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_3 $}"+" & "+str(round(means_wg_rg[2][0].item(),4)) + "$\pm$"+str(round(stds_wg_rg[2][0].item(),4))+ " & "+str(round(means_wg_rg[2][1].item(),4)) + "$\pm$"+str(round(stds_wg_rg[2][1].item(),4))+  " & "+str(round(means_wg_rg[2][2].item(),4)) + "$\pm$"+str(round(stds_wg_rg[2][2].item(),4))+ " & "+str(round(means_wg_rg[2][3].item(),4)) + "$\pm$"+str(round(stds_wg_rg[2][3].item(),4))+" \\\\\n"
    # latex_code += "\\hline\n"
    latex_code += "{}"+"&"+"\multicolumn{1}{c}{$x^{cf}_4 $}"+" & "+  str(round(means_wg_rg[3][0].item(),4)) + "$\pm$"+str(round(stds_wg_rg[3][0].item(),4))+ " & "+str(round(means_wg_rg[3][1].item(),4)) + "$\pm$"+str(round(stds_wg_rg[3][1].item(),4))+  " & "+str(round(means_wg_rg[3][2].item(),4)) + "$\pm$"+str(round(stds_wg_rg[3][2].item(),4))+ " & "+str(round(means_wg_rg[3][3].item(),4)) + "$\pm$"+str(round(stds_wg_rg[3][3].item(),4))+" \\\\\n"
    latex_code += "\\bottomrule\n"

    latex_code += "\\end{tabular}\n"
    latex_code +="\\caption{}\n"
    latex_code +="\\label{}\n"
    latex_code +="\\end{table}"
    return latex_code

# def count_mean_std(X:Tensor,Y:Tensor) -> Tensor:
def plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_mismatched,i,features: [],file:str):

    df = preparator_data.create_df([x_cf[f"{i}"], x_cf_real_graph[f"{i}"],x_cf_mismatched[f"{i}"]], ["Ground truth", "Reference","Mismatched"])
    df_melted = pd.melt(df, id_vars=['mode'], var_name='$x_{cf}$', value_name='value')
    # df_melted =df_melted.rename(columns={'mode':'counterfactual'})

    
    sns.set(font_scale=1.08, style='ticks', context='paper', palette='tab10')
    df_plot   = df_melted[df_melted['$x_{cf}$'].str.contains('|'.join(features))]
    # plt.rc({'text.usetex': True, 'font.family': 'serif', 'text.latex.preamble': '\\usepackage{times} ', 'figure.figsize': (3.25, 2.0086104634371584), 'figure.constrained_layout.use': True, 'figure.autolayout': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015, 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.titlesize': 8})
    g = sns.FacetGrid(df_plot, col="$x_{cf}$", hue="mode", sharey=False, col_wrap=1, height=2.008, aspect=1.5)
    
    # plt.rc('font', family='serif', serif=['Computer Modern Roman'])
    # plt.rc('text', usetex=True) 
    # Map the same plot type to each subplot
    g.map(sns.kdeplot, "value", common_norm=False, fill= False, alpha=0.6, linewidth=2)
    # legend = g.get_legend(loc='upper right')
    # legend.set_title("counterfactual", prop={'size': 11})
    # plt.setp(legend.get_texts(), fontsize=11)
    # g.set_xlabels("Value",fontsize=11)
    # g.set_ylabels("Density",fontsize=11)
    # g.set_titles(title='counterfactual',fontsize=11)
    # Add a legend
    g.add_legend(loc='upper right', title='Model') #, fontsize=7.0)
    # plt.xticks(fontsize=11)
    # plt.yticks(fontsize=11)
    # plt.setp("x_cf",fontsize=11)
    plt.savefig(file, dpi=200)
    # Show the plot
    plt.show()

    return 


my_index = torch.tensor(range(0,len(x[0])))
##counterfactual error
## configurate the models with different causal graphs
model_real_graph = models(cfg, cfg.dataset,'simpson_nonlinear_seed_3')

model_add_x1tox4 = models(cfg, cfg.dataset,'simpson_nonlinear_x1tox4')
model_add_x2tox4 = models(cfg, cfg.dataset,'simpson_nonlinear_x2tox4')

model_remove_x1tox3 = models(cfg, cfg.dataset,'simpson_nonlinear_x1tox3')
model_remove_x2tox3 = models(cfg, cfg.dataset,'simpson_nonlinear_x2tox3_seed_3')
model_remove_x3tox4 = models(cfg, cfg.dataset,'simpson_nonlinear_x3tox4_seed_3')
model_remove_x1tox2 = models(cfg, cfg.dataset,'simpson_nonlinear_x1tox2_seed_3')

model_change_x2tox1 = models(cfg, cfg.dataset,'simpson_nonlinear_change_x2tox1_seed_2')
model_change_x3tox2 = models(cfg, cfg.dataset,'simpson_nonlinear_change_x3tox2_seed_3')
model_change_x4tox3 = models(cfg, cfg.dataset,'simpson_nonlinear_change_x4tox3_seed_3')

# ground_truth model
functions, inverses= get_ground_truth_SCM('simpson')
# get the exogenous variables
u_obs=update_U(x,inverses)
#base line

x_cf = {}
x_cf_real_graph={}
x_cf_add_x1tox4={}
x_cf_add_x2tox4={}
x_cf_remove_x1tox3={}
x_cf_remove_x2tox3={}
x_cf_remove_x3tox4={}
x_cf_remove_x1tox2={}
x_cf_change_x2tox1={}
x_cf_change_x3tox2={}
x_cf_change_x4tox3={}


error_baseline_mean={}
error_baseline_std={}
# error_mean ={}
## add with the ground truth
error_mean_add_x2tox4={}
error_mean_add_x1tox4={}
## remove with the ground truth
error_mean_remove_x1tox3={}
error_mean_remove_x2tox3={}
error_mean_remove_x3tox4={}
error_mean_remove_x1tox2={}
## change direction with the ground truth
error_mean_change_x2tox1={}
error_mean_change_x3tox2={}
error_mean_change_x4tox3={}

## std
error_std_add_x2tox4={}
error_std_add_x1tox4={}

error_std_remove_x1tox3={}
error_std_remove_x2tox3={}
error_std_remove_x3tox4={}
error_std_remove_x1tox2={}

error_std_change_x2tox1={}
error_std_change_x3tox2={}
error_std_change_x4tox3={}

##Check the direction
error_mean_add_x1tox4_real_graph={}
error_mean_add_x2tox4_real_graph={}

error_std_add_x1tox4_real_graph={}
error_std_add_x2tox4_real_graph={}
## Remove
error_mean_remove_x1tox3_real_graph={}
error_mean_remove_x2tox3_real_graph={}
error_mean_remove_x3tox4_real_graph={}
error_mean_remove_x1tox2_real_graph={}

error_std_remove_x1tox3_real_graph={}
error_std_remove_x2tox3_real_graph={}
error_std_remove_x3tox4_real_graph={}
error_std_remove_x1tox2_real_graph={}
##Change
error_mean_change_x2tox1_real_graph={}
error_mean_change_x3tox2_real_graph={}
error_mean_change_x4tox3_real_graph={}

error_std_change_x2tox1_real_graph={}
error_std_change_x3tox2_real_graph={}
error_std_change_x4tox3_real_graph={}

error_std ={}

import seaborn as sns

for index in my_index:
    x_tmp_1 = x.clone()
    x_cf_index=ground_counterfactual(x_tmp_1,u_obs,index,interven_value[index],functions,inverses)
    # x_cf[f"{index}"]=x_cf_index
    x_tmp_2 = x.clone()
    x_cf_real_graph_index= model_real_graph.compute_counterfactual(
                    x_tmp_2, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    ## normalized by hand
    # x_tmp_2_norm = scaler.transform(x_tmp_2, inplace = False) ## x normalized
    # interven_value_norm = torch.div((interven_value[index]-x_mean[index]),x_std[index]) # interven_value normalized

    # x_cf_real_graph_index_2= model_real_graph.compute_counterfactual(
    #                 x_tmp_2_norm, index, value=interven_value_norm,scaler = None

    # )
    # x_cf_real_graph_index_2=scaler.inverse_transform(x_cf_real_graph_index_2) ## back to obs
    # x_cf_real_graph[f"{index}"]=x_cf_real_graph_index
    # x_cf_real_graph_2[f"{index}"]=x_cf_real_graph_index_2
    ##Add
   

    x_tmp_3 = x.clone()
    x_cf_add_x1tox4_index = model_add_x1tox4.compute_counterfactual(
                    x_tmp_3, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    # x_tmp_3_norm = scaler.transform(x_tmp_3, inplace = False) ## x normalized
    # # interven_value_norm = torch.div((interven_value[index]-x_mean[index]),x_std[index]) # interven_value normalized

    # x_cf_add_x1tox4_index_2= model_add_x1tox4.compute_counterfactual(
    #                 x_tmp_3_norm, index, value=interven_value_norm,scaler = None

    # )
    # x_cf_add_x1tox4_index_2=scaler.inverse_transform(x_cf_add_x1tox4_index_2) ## back to obs
    # x_cf_add_x1tox4[f"{index}"]=x_cf_add_x1tox4_index
    # x_cf_add_x1tox4_2[f"{index}"]=x_cf_add_x1tox4_index_2


    x_tmp_4 = x.clone()
    x_cf_add_x2tox4_index = model_add_x2tox4.compute_counterfactual(
                    x_tmp_4, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    ## Remove
    x_tmp_5 = x.clone()
    x_cf_remove_x1tox3_index = model_remove_x1tox3.compute_counterfactual(
                    x_tmp_5, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    x_tmp_6 = x.clone()
    x_cf_remove_x2tox3_index = model_remove_x2tox3.compute_counterfactual(
                    x_tmp_6, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )

    x_tmp_7 = x.clone()
    x_cf_remove_x3tox4_index = model_remove_x3tox4.compute_counterfactual(
                    x_tmp_7, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    x_tmp_8 = x.clone()
    x_cf_remove_x1tox2_index = model_remove_x1tox2.compute_counterfactual(
                    x_tmp_8, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    ## Change
    x_tmp_9 = x.clone()
    x_cf_change_x2tox1_index = model_change_x2tox1.compute_counterfactual(
                    x_tmp_9, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    x_tmp_10 = x.clone()
    x_cf_change_x3tox2_index = model_change_x3tox2.compute_counterfactual(
                    x_tmp_10, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )
    x_tmp_11 = x.clone()
    x_cf_change_x4tox3_index = model_change_x4tox3.compute_counterfactual(
                    x_tmp_11, index, value=interven_value[index],scaler = preparator_data.scaler_transform

    )

    counterfactual_error_mean_baseline = torch.mean(torch.sqrt(torch.sub(x_cf_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_baseline = torch.std(torch.sqrt(torch.sub(x_cf_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean_add_x1tox4=torch.mean(torch.sqrt(torch.sub(x_cf_add_x1tox4_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_add_x1tox4=torch.std(torch.sqrt(torch.sub(x_cf_add_x1tox4_index,x_cf_index) ** 2), axis=0)

    counterfactual_error_mean_add_x2tox4=torch.mean(torch.sqrt(torch.sub(x_cf_add_x2tox4_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_add_x2tox4=torch.std(torch.sqrt(torch.sub(x_cf_add_x2tox4_index,x_cf_index) ** 2), axis=0)

    counterfactual_error_mean_remove_x1tox3=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x1tox3_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_remove_x1tox3=torch.std(torch.sqrt(torch.sub(x_cf_remove_x1tox3_index,x_cf_index) ** 2), axis=0)

    counterfactual_error_mean_remove_x2tox3=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x2tox3_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_remove_x2tox3=torch.std(torch.sqrt(torch.sub(x_cf_remove_x2tox3_index,x_cf_index) ** 2), axis=0)

    counterfactual_error_mean_remove_x3tox4=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x3tox4_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_remove_x3tox4=torch.std(torch.sqrt(torch.sub(x_cf_remove_x3tox4_index,x_cf_index) ** 2), axis=0)

    counterfactual_error_mean_remove_x1tox2=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x1tox2_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_remove_x1tox2=torch.std(torch.sqrt(torch.sub(x_cf_remove_x1tox2_index,x_cf_index) ** 2), axis=0)

    counterfactual_error_mean_change_x2tox1=torch.mean(torch.sqrt(torch.sub(x_cf_change_x2tox1_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_change_x2tox1=torch.std(torch.sqrt(torch.sub(x_cf_change_x2tox1_index,x_cf_index) ** 2), axis=0)

    counterfactual_error_mean_change_x3tox2=torch.mean(torch.sqrt(torch.sub(x_cf_change_x3tox2_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_change_x3tox2=torch.std(torch.sqrt(torch.sub(x_cf_change_x3tox2_index,x_cf_index) ** 2), axis=0)
    
    counterfactual_error_mean_change_x4tox3=torch.mean(torch.sqrt(torch.sub(x_cf_change_x4tox3_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std_change_x4tox3=torch.std(torch.sqrt(torch.sub(x_cf_change_x4tox3_index,x_cf_index) ** 2), axis=0)


    # compare with the real graph
    counterfactual_error_mean_remove_x1tox3_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x1tox3_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_remove_x1tox3_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_remove_x1tox3_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean_remove_x3tox4_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x3tox4_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_remove_x3tox4_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_remove_x3tox4_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean_remove_x1tox2_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x1tox2_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_remove_x1tox2_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_remove_x1tox2_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean_remove_x2tox3_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_remove_x2tox3_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_remove_x2tox3_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_remove_x2tox3_index,x_cf_real_graph_index) ** 2), axis=0)
    

    counterfactual_error_mean_add_x1tox4_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_add_x1tox4_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_add_x1tox4_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_add_x1tox4_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean_add_x2tox4_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_add_x2tox4_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_add_x2tox4_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_add_x2tox4_index,x_cf_real_graph_index) ** 2), axis=0)


    counterfactual_error_mean_change_x2tox1_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_change_x2tox1_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_change_x2tox1_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_change_x2tox1_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean_change_x3tox2_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_change_x3tox2_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_change_x3tox2_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_change_x3tox2_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean_change_x4tox3_real_graph=torch.mean(torch.sqrt(torch.sub(x_cf_change_x4tox3_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_change_x4tox3_real_graph=torch.std(torch.sqrt(torch.sub(x_cf_change_x4tox3_index,x_cf_real_graph_index) ** 2), axis=0)


    # counterfactual_error_std = torch.std(torch.sqrt(torch.sub(x_cf_wrong_graph_index, x_cf_index) ** 2),axis=0)
    error_baseline_mean[f"{index}"] = counterfactual_error_mean_baseline
    error_baseline_std[f"{index}"] = counterfactual_error_std_baseline

    error_mean_add_x1tox4[f"{index}"] = counterfactual_error_mean_add_x1tox4
    error_std_add_x1tox4[f"{index}"] = counterfactual_error_std_add_x1tox4

    error_mean_add_x2tox4[f"{index}"] = counterfactual_error_mean_add_x2tox4
    error_std_add_x2tox4[f"{index}"] = counterfactual_error_std_add_x2tox4

    error_mean_remove_x1tox3[f"{index}"] = counterfactual_error_mean_remove_x1tox3
    error_std_remove_x1tox3[f"{index}"] = counterfactual_error_std_remove_x1tox3

    error_mean_remove_x2tox3[f"{index}"] = counterfactual_error_mean_remove_x2tox3
    error_std_remove_x2tox3[f"{index}"] = counterfactual_error_std_remove_x2tox3

    error_mean_remove_x3tox4[f"{index}"] = counterfactual_error_mean_remove_x3tox4
    error_std_remove_x3tox4[f"{index}"] = counterfactual_error_std_remove_x3tox4

    error_mean_remove_x1tox2[f"{index}"] = counterfactual_error_mean_remove_x1tox2
    error_std_remove_x1tox2[f"{index}"] = counterfactual_error_std_remove_x1tox2
    
    error_mean_change_x2tox1[f"{index}"] = counterfactual_error_mean_change_x2tox1
    error_std_change_x2tox1[f"{index}"] = counterfactual_error_std_change_x2tox1
    
    error_mean_change_x3tox2[f"{index}"] = counterfactual_error_mean_change_x3tox2
    error_std_change_x3tox2[f"{index}"] = counterfactual_error_std_change_x3tox2
    
    error_mean_change_x4tox3[f"{index}"] = counterfactual_error_mean_change_x4tox3
    error_std_change_x4tox3[f"{index}"] = counterfactual_error_std_change_x4tox3

    
    error_mean_add_x1tox4_real_graph[f"{index}"]=counterfactual_error_mean_add_x1tox4_real_graph
    error_std_add_x1tox4_real_graph[f"{index}"]=counterfactual_error_std_add_x1tox4_real_graph

    error_mean_add_x2tox4_real_graph[f"{index}"]=counterfactual_error_mean_add_x2tox4_real_graph
    error_std_add_x2tox4_real_graph[f"{index}"]=counterfactual_error_std_add_x2tox4_real_graph

    error_mean_remove_x1tox2_real_graph[f"{index}"]=counterfactual_error_mean_remove_x1tox2_real_graph
    error_std_remove_x1tox2_real_graph[f"{index}"]=counterfactual_error_std_remove_x1tox2_real_graph

    error_mean_remove_x2tox3_real_graph[f"{index}"]=counterfactual_error_mean_remove_x2tox3_real_graph
    error_std_remove_x2tox3_real_graph[f"{index}"]=counterfactual_error_std_remove_x2tox3_real_graph

    error_mean_remove_x3tox4_real_graph[f"{index}"]=counterfactual_error_mean_remove_x3tox4_real_graph
    error_std_remove_x3tox4_real_graph[f"{index}"]=counterfactual_error_std_remove_x3tox4_real_graph

    error_mean_remove_x1tox3_real_graph[f"{index}"]=counterfactual_error_mean_remove_x1tox3_real_graph
    error_std_remove_x1tox3_real_graph[f"{index}"]=counterfactual_error_std_remove_x1tox3_real_graph

    error_mean_change_x2tox1_real_graph[f"{index}"]=counterfactual_error_mean_change_x2tox1_real_graph
    error_std_change_x2tox1_real_graph[f"{index}"]=counterfactual_error_std_change_x2tox1_real_graph

    error_mean_change_x3tox2_real_graph[f"{index}"]=counterfactual_error_mean_change_x3tox2_real_graph
    error_std_change_x3tox2_real_graph[f"{index}"]=counterfactual_error_std_change_x3tox2_real_graph

    error_mean_change_x4tox3_real_graph[f"{index}"]=counterfactual_error_mean_change_x4tox3_real_graph
    error_std_change_x4tox3_real_graph[f"{index}"]=counterfactual_error_std_change_x4tox3_real_graph
    # error_std[f"{index}"] = counterfactual_error_std

    x_cf[f"{index}"]=x_cf_index
    x_cf_real_graph[f"{index}"]=x_cf_real_graph_index
    x_cf_add_x1tox4[f"{index}"]=x_cf_add_x1tox4_index
    x_cf_add_x2tox4[f"{index}"]=x_cf_add_x2tox4_index
    x_cf_remove_x1tox3[f"{index}"]=x_cf_remove_x1tox3_index
    x_cf_remove_x2tox3[f"{index}"]=x_cf_remove_x2tox3_index
    x_cf_remove_x3tox4[f"{index}"]=x_cf_remove_x3tox4_index
    x_cf_remove_x1tox2[f"{index}"]=x_cf_remove_x1tox2_index
    x_cf_change_x2tox1[f"{index}"]=x_cf_change_x2tox1_index
    x_cf_change_x3tox2[f"{index}"]=x_cf_change_x3tox2_index
    x_cf_change_x4tox3[f"{index}"]=x_cf_change_x4tox3_index




# Convert dictionary to LaTeX table



##Output the experiments results
# x_norm = scaler.transform(x, inplace = False) ## x normalized
# print('x to u')
# print(torch.mean(y,axis=0))
# print(torch.mean(model_real_graph.flow().transform(x_norm), axis=0))

# print(torch.mean(model_add_x1tox4.flow().transform(x_norm), axis=0))
# print(torch.mean(model_add_x2tox4.flow().transform(x_norm), axis=0))

# print(torch.mean(model_remove_x1tox3.flow().transform(x_norm), axis=0))
# print(torch.mean(model_remove_x2tox3.flow().transform(x_norm), axis=0))
# print(torch.mean(model_remove_x3tox4.flow().transform(x_norm), axis=0))
# print(torch.mean(model_remove_x1tox2.flow().transform(x_norm), axis=0))

# print(torch.mean(model_change_x2tox1.flow().transform(x_norm), axis=0))
# print(torch.mean(model_change_x3tox2.flow().transform(x_norm), axis=0))
# print(torch.mean(model_change_x4tox3.flow().transform(x_norm), axis=0))
# print('x to u std')

# print('u to x')
# print(torch.mean(x,axis=0))
# print(torch.mean(scaler.inverse_transform(model_real_graph.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_add_x1tox4.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_add_x2tox4.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_remove_x1tox3.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_remove_x2tox3.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_remove_x3tox4.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_remove_x1tox2.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_change_x2tox1.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_change_x3tox2.flow().transform.inv(y)), axis=0))
# print(torch.mean(scaler.inverse_transform(model_change_x4tox3.flow().transform.inv(y)), axis=0))

# print('\subsubsection{x1 to x4}')
# latex_table_add_x1tox4= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_add_x1tox4,error_std_add_x1tox4,error_mean_add_x1tox4_real_graph,error_std_add_x1tox4_real_graph)
# print(latex_table_add_x1tox4)


# print('\subsubsection{x2 to x4}')
# latex_table_add_x2tox4= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_add_x2tox4,error_std_add_x2tox4,error_mean_add_x2tox4_real_graph,error_std_add_x2tox4_real_graph)
# print(latex_table_add_x2tox4)


# print('\subsubsection{x1 to x3}')
# latex_table_remove_x1tox3= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_remove_x1tox3,error_std_remove_x1tox3 ,error_mean_remove_x1tox3_real_graph,error_std_remove_x1tox3_real_graph)
# print(latex_table_remove_x1tox3)

# print('\subsubsection{x2 to x3}')
# latex_table_remove_x2tox3= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_remove_x2tox3,error_std_remove_x2tox3 ,error_mean_remove_x2tox3_real_graph,error_std_remove_x2tox3_real_graph)
# print(latex_table_remove_x2tox3)

# print('\subsubsection{x3 to x4}')
# latex_table_remove_x3tox4= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_remove_x3tox4,error_std_remove_x3tox4 ,error_mean_remove_x3tox4_real_graph,error_std_remove_x3tox4_real_graph)
# print(latex_table_remove_x3tox4)



# print('\subsubsection{x1 to x2}')
# latex_table_remove_x1tox2= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_remove_x1tox2,error_std_remove_x1tox2 ,error_mean_remove_x1tox2_real_graph,error_std_remove_x1tox2_real_graph)
# print(latex_table_remove_x1tox2)

# print('\subsubsection{x2 to x1}')
# latex_table_change_x2tox1= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_change_x2tox1,error_std_change_x2tox1 ,error_mean_change_x2tox1_real_graph,error_std_change_x2tox1_real_graph)
# print(latex_table_change_x2tox1)



# print('\subsubsection{x3 to x2}')
# latex_table_change_x3tox2= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_change_x3tox2,error_std_change_x3tox2 ,error_mean_change_x3tox2_real_graph,error_std_change_x3tox2_real_graph)
# print(latex_table_change_x3tox2)

# print('\subsubsection{x4 to x3}')
# latex_table_change_x4tox3= dict_to_latex_table(error_baseline_mean,error_baseline_std,error_mean_change_x4tox3,error_std_change_x4tox3 ,error_mean_change_x4tox3_real_graph,error_std_change_x4tox3_real_graph)
# print(latex_table_change_x4tox3)


# plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_add_x1tox4,0,['4'],file="Add_x_1tox_4,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_add_x1tox4,1,['4'],file="Add_x_1tox_4,do(x_2=4).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_add_x1tox4,2,['4'],file="Add_x_1tox_4,do(x_3=5).png")

# plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_add_x2tox4,0,['4'],file="Add_x_2tox_4,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_add_x2tox4,1,['4'],file="Add_x_2tox_4,do(x_2=4).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_add_x2tox4,2,['4'],file="Add_x_2tox_4,do(x_3=5).png")

# plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x1tox3,0,['3','4'],file="Remove_x_1tox_3,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x1tox3,1,['3','4'],file="Remove_x_1tox_3,do(x_2=4).png")

# plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x2tox3,0,['3','4'],file="Remove_x_2tox_3,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x2tox3,1,['3','4'],file="Remove_x_2tox_3,do(x_2=4).png")

# plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x3tox4,0,['4'],file="Remove_x_3tox_4,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x3tox4,1,['4'],file="Remove_x_3tox_4,do(x_2=4).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x3tox4,2,['4'],file="Remove_x_3tox_4,do(x_3=5).png")

# plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_remove_x1tox2,0,['2','3','4'],file="Remove_x_1tox_2,do(x_1=2.5).png")

#plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x2tox1,0,['2','3','4'],file="Change_x_2tox_1,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x2tox1,1,['1','3','4'],file="Change_x_2tox_1,do(x_2=4).png")

#plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x3tox2,0,['2','3','4'],file="Change_x_3tox_2,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x3tox2,1,['3','4'],file="Change_x_3tox_2,do(x_2=4).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x3tox2,2,['2'],file="Change_x_3tox_2,do(x_3=5).png")

#plot
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x4tox3,0,['3','4'],file="Change_x_4tox_3,do(x_1=2.5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x4tox3,1,['3','4'],file="Change_x_4tox_3,do(x_2=4).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x4tox3,2,['4'],file="Change_x_4tox_3,do(x_3=5).png")
plot_counterfactual_inference(x_cf,x_cf_real_graph,x_cf_change_x4tox3,3,['3'],file="Change_x_4tox_3,do(x_4=4).png")