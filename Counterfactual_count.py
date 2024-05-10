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

# make a config file
args_list, args = causal_nf_config.parse_args()

config = causal_nf_config.build_config(
    config_file=args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)

# prepare data
preparator_data = SCMPreparator.loader(cfg.dataset)
preparator_data.prepare_data()
loader = preparator_data._data_loader(
        preparator_data.datasets[0], batch_size=len(preparator_data.datasets[0]), shuffle=False, num_workers=0
    )
x, y = next((iter(loader)))

interven_value = torch.tensor([-2,-2,-2,-2])

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
                    x, index, value=interven_value[index],scaler = None

    ) ## counterfactual for the real graph
   output_cf_wrong_index = model2.compute_counterfactual(
        x, index, value=interven_value[index],scaler = None
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


my_index = torch.tensor(range(0,len(x[0])))
##counterfactual error
## configurate the models with different causal graphs
model_real_graph = models(cfg, cfg.dataset,'simpson_nonlinear_50000')
model_2 = models(cfg, cfg.dataset2,'simpson_nonlinear_x1tox2')
# ground_truth model
functions, inverses= get_ground_truth_SCM('simpson')
# get the exogenous variables
u_obs=update_U(x,inverses)
#base line

# x_cf = {}
error_baseline_mean={}
error_baseline_std={}
error_mean ={}
error_std ={}
for index in my_index:
    x_tmp = x.clone()
    x_cf_index=ground_counterfactual(x,u_obs,index,interven_value[index],functions,inverses)
    # x_cf[f"{index}"]=x_cf_index
    x_tmp = x.clone()
    x_cf_real_graph_index= model_real_graph.compute_counterfactual(
                    x_tmp, index, value=interven_value[index],scaler = None

    )
    x_tmp = x.clone()
    x_cf_wrong_graph_index = model_2.compute_counterfactual(
                    x_tmp, index, value=interven_value[index],scaler = None

    )
    counterfactual_error_mean_baseline = torch.mean(torch.sqrt(torch.sub(x_cf_index,x_cf_real_graph_index) ** 2), axis=0)
    counterfactual_error_std_baseline = torch.std(torch.sqrt(torch.sub(x_cf_index,x_cf_real_graph_index) ** 2), axis=0)

    counterfactual_error_mean=torch.mean(torch.sqrt(torch.sub(x_cf_wrong_graph_index,x_cf_index) ** 2), axis=0)
    counterfactual_error_std = torch.std(torch.sqrt(torch.sub(x_cf_wrong_graph_index, x_cf_index) ** 2),axis=0)
    error_baseline_mean[f"{index}"] = counterfactual_error_mean_baseline
    error_baseline_std[f"{index}"] = counterfactual_error_std_baseline
    error_mean[f"{index}"] = counterfactual_error_mean
    error_std[f"{index}"] = counterfactual_error_std


#print(counter_error)
print(error_baseline_mean)
print(error_mean)
# print(counter_error_std)






#### Code reference

## Counterfactual inference with method 1
# import numpy as np
# u_obs_1 = torch.zeros(50000,4)
# x_cf_1 = torch.zeros(50000,4)
# ## sym-prob
# functions = [
#                 lambda u1: u1,
#                 lambda x1, u2: 2 * torch.tanh(2 * x1) + 1 / np.sqrt(10) * u2,
#                 lambda x1, x2, u3: 1 / 2 * x1 * x2 + 1 / np.sqrt(2) * u3,
#                 lambda x1, x2, x3, u4: torch.tanh(3 / 2 * x1) + np.sqrt(3 / 10) * u4,
#             ]
#
# inverses = [
#     lambda x1: x1,
#     lambda x1, x2: np.sqrt(10) * (x2 - 2 * torch.tanh(2 * x1)),
#     lambda x1, x2, x3: np.sqrt(2) * (x3 - 1 / 2 * x1 * x2),
#     lambda x1, x2, x3, x4: 1
#                            / np.sqrt(3 / 10)
#                            * (x4 - torch.tanh(3 / 2 * x1)),
# ]

# ##non-linear
# s = torch.nn.functional.softplus
#
# functions_non = [
#                 lambda u1: u1,
#                 lambda x1, u2: s(1.0 - x1) + np.sqrt(3 / 20.0) * u2,
#                 lambda x1, x2, u3: torch.tanh(2 * x2) + 3 / 2 * x1 - 1 + torch.tanh(u3),
#                 lambda x1, x2, x3, u4: (x3 - 4.0) / 5.0 + 3 + 1 / np.sqrt(10) * u4,
#             ]
#
# functions_unroll = [
        #     lambda u1: u1,
        #     lambda u1, u2: s(1.0 - functions[0](u1)) + np.sqrt(3 / 20.0) * u2,
        #     lambda u1, u2, u3: torch.tanh(2 * functions[1](functions[0](u1),
        #                                                    u2)) + 3 / 2 * u1 - 1 + torch.tanh(u3),
        #     lambda u1, u2, u3, u4: (functions[2](functions[0](u1),
        #                                          functions[1](functions[0](u1), u2),
        #                                          u3) - 4.0) / 5.0 + 3 + 1 / np.sqrt(10) * u4,
        # ]
# inverses_non = [
#                 lambda x1: x1,
#                 lambda x1, x2: 1 / np.sqrt(3 / 20.0) * (x2 - s(1.0 - x1)),
#                 lambda x1, x2, x3: torch.atanh(
#                     x3 - torch.tanh(2 * x2) - 3 / 2 * x1 + 1
#                 ),
#                 lambda _, __, x3, x4: np.sqrt(10) * (x4 - (x3 - 4.0) / 5.0 - 3),
#             ]
# for i in range(0,50000):
# # get the u
#    u_obs_1[i]=torch.tensor([inverses_non[0](x[i][0]),
#                           inverses_non[1](x[i][0],x[i][1]),
#                           inverses_non[2](x[i][0],x[i][1],x[i][2]),
#                           inverses_non[3](x[i][0],x[i][1],x[i][2],x[i][3])])
#    # x[i][0]=interven_value[0]


 # if 'simpson' in config["dataset__name"]:
    #     functions = sem_equations.Simpson(config["dataset__sem_name"]).functions
    #     inverses = sem_equations.Simpson(config["dataset__sem_name"]).inverses
    #     s = torch.nn.functional.softplus
    #     functions_unroll = [
    #         lambda u1: u1,
    #         lambda u1, u2: s(1.0 - functions[0](u1)) + np.sqrt(3 / 20.0) * u2,
    #         lambda u1, u2, u3: torch.tanh(2 * functions[1](functions[0](u1),
    #                                                        u2)) + 3 / 2 * u1 - 1 + torch.tanh(u3),
    #         lambda u1, u2, u3, u4: (functions[2](functions[0](u1),
    #                                              functions[1](functions[0](u1), u2),
    #                                              u3) - 4.0) / 5.0 + 3 + 1 / np.sqrt(10) * u4,
    #     ]

##### counterfactual with method 2
 # get the u
        # u_obs[i] = torch.tensor([inverses[0](X_factual[i][0]),
        #                          inverses[1](X_factual[i][0], X_factual[i][1]),
        #                          inverses[2](X_factual[i][0], X_factual[i][1], X_factual[i][2]),
        #                          inverses[3](X_factual[i][0], X_factual[i][1], X_factual[i][2], X_factual[i][3])])
        # ## how to call the lambda functions

        # do intervention
        #
        # u_temp[i] = torch.tensor([inverses[0](X_factual[i][0]),
        #                           inverses[1](X_factual[i][0], X_factual[i][1]),
        #                           inverses[2](X_factual[i][0], X_factual[i][1], X_factual[i][2]),
        #                           inverses[3](X_factual[i][0], X_factual[i][1], X_factual[i][2], X_factual[i][3])])
        #u_obs[:, index] = u_temp[:, index]
        # U_obs[i][index] = u_temp[i][index]
        # x_cf[i] = torch.tensor([functions_unroll[0](U_obs[i][0]),
        #                         functions_unroll[1](U_obs[i][0], U_obs[i][1]),
        #                         functions_unroll[2](U_obs[i][0], U_obs[i][1], U_obs[i][2]),
        #                         functions_unroll[3](U_obs[i][0], U_obs[i][1], U_obs[i][2], U_obs[i][3])])
        # has_nan = torch.isnan(x_cf[i]).any().item()

##### calculation
# for i in range(0,50000):

# counter_error = torch.mean(torch.sqrt(torch.sub(x_cf,output_cf["0"])**2),axis=0)
# counter_error_std = torch.std(torch.sqrt(torch.sub(x_cf,output_cf["0"])**2),axis=0)
#
# ## method 2
# u_obs = torch.zeros(50000,4)
# u_temp = torch.zeros(50000,4)
# x_cf_2 = torch.zeros(50000,4)
# for i in range(0,50000):
#     u_obs[i] = torch.tensor([inverses_non[0](x[i][0]),
#                              inverses_non[1](x[i][0], x[i][1]),
#                              inverses_non[2](x[i][0], x[i][1], x[i][2]),
#                              inverses_non[3](x[i][0], x[i][1], x[i][2], x[i][3])])
#     x[i][0] = interven_value[0]
#     u_temp[i]= torch.tensor([inverses_non[0](x[i][0]),
#                              inverses_non[1](x[i][0], x[i][1]),
#                              inverses_non[2](x[i][0], x[i][1], x[i][2]),
#                              inverses_non[3](x[i][0], x[i][1], x[i][2], x[i][3])])
#     u_obs[i][0]=u_temp[i][0]
#     x_cf_2[i] = torch.tensor([functions_non_unroll[0](u_obs[i][0]),
#                               functions_non_unroll[1](u_obs[i][0], u_obs[i][1]),
#                               functions_non_unroll[2](u_obs[i][0], u_obs[i][1], u_obs[i][2]),
#                               functions_non_unroll[3](u_obs[i][0],u_obs[i][1],u_obs[i][2], u_obs[i][3])])
# counter_error = torch.mean(torch.sqrt(torch.sub(x_cf,x_cf_2)**2),axis=0)

##### Test with one index
# index =1
# x_cf_index=ground_counterfactual(x,u_obs,index,interven_value[index],functions,inverses)
# x_tmp = x.clone()
# x_cf_wrong_graph_index = model_2.compute_counterfactual(
#                     x_tmp, index, value=interven_value[index],scaler = None

#     )
# x_tmp_2 = x.clone()
# x_cf_real_graph_index= model_real_graph.compute_counterfactual(
#                     x_tmp_2, index, value=interven_value[index],scaler = None

#     )
# counterfactual_error_mean_baseline = torch.mean(torch.sqrt(torch.sub(x_cf_index,x_cf_real_graph_index) ** 2), axis=0)
# counterfactual_error_mean=torch.mean(torch.sqrt(torch.sub(x_cf_index,x_cf_wrong_graph_index) ** 2), axis=0)
# print(counterfactual_error_mean)
# print(counterfactual_error_mean_baseline)
# for i in range(0,50000):
#     x_cf_1[i] = torch.tensor([interven_value[0],
#                               functions_non[1](x[i][0], u_obs_1[i][1]),
#                               functions_non[2](x[i][0], functions_non[1](x[i][0], u_obs_1[i][1]), u_obs_1[i][2]),
#                               functions_non[3](x[i][0], functions_non[1](x[i][0], u_obs_1[i][1]),
#                                                functions_non[2](x[i][0], functions_non[1](x[i][0], u_obs_1[i][1]),
#                                                                 u_obs_1[i][2]), u_obs_1[i][3])])


# counter_error_std = torch.std(torch.sqrt(torch.sub(x_cf_1,x_cf)**2),axis=0)

