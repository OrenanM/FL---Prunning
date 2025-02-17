# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientala import clientALA
from flcore.servers.serverbase import Server
from threading import Thread

from utils.prunning import restore_to_original_size, prune_and_restructure
from utils.size_mode import get_model_size
from utils.prunning_nisp import prune_fc1
from utils.prunning_snip import snip_pruning, apply_mask
import copy


class FedALA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientALA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.current_round = i
            self.selected_clients = self.select_clients()

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            t_send = time.time() - s_t # tempo de envio

            s_train = time.time()
            '''threads = [Thread(target=client.train)
                       for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]'''

            for client in self.selected_clients:
                client.train()
            t_train = time.time() - s_train # tempo de treinamento
            t_train = t_train if self.time_threthold > t_train else self.time_threthold
            self.time_train.append(t_train)

            st_agr = time.time()
            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            t_aggregate = time.time() - st_agr
            
            tot_time = t_aggregate + t_train + t_send
            self.Budget.append(tot_time)
            self.n_aggregates.append(len(self.uploaded_ids))

            print(f'time train: {self.time_train[-1]}')
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientALA)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self):
        assert len(self.clients) > 0

        # Cálculo inicial do pruning apenas na primeira rodada, se aplicável
        max_amount = None
        if self.current_round == 1 and self.pruning_method:
            max_amount = self.set_amount_prune()
            
        # Exibir tamanho do modelo global
        size_global_model = get_model_size(self.global_model)
        print(f'Size Global Model: {size_global_model:.2f}MB')

        # Copia profunda do modelo global
        global_model_copy = copy.deepcopy(self.global_model)

        # Processar os clientes
        for client in self.clients:
            start_time = time.time()
            g_model_pruned = copy.deepcopy(self.global_model)
            local_model=client.model

            if self.current_round == 1 and self.pruning_method:
                
                if self.pruning_method == 'OPALA':
                    g_model_pruned, _ = prune_and_restructure(model=self.global_model, 
                                                              pruning_rate=max_amount, 
                                                              size_fc=self.size_fc)
                    local_model, _ = prune_and_restructure(model=client.model, 
                                                           pruning_rate=max_amount, 
                                                           size_fc=self.size_fc)
                elif self.pruning_method == 'NISP':
                    trainloader = client.load_train_data()
                    
                    g_model_pruned, _ = prune_fc1(model=client.model, 
                                                       dataloader=trainloader, 
                                                       pruning_ratio=max_amount,
                                                       device=self.device)
                    
                    local_model, _ = prune_fc1(model=client.model, 
                                               dataloader=trainloader, 
                                               pruning_ratio=max_amount,
                                               device=self.device)
                
                elif self.pruning_method == 'SNIP':
                    trainloader = client.load_train_data()
                    
                    self.mask = snip_pruning(model=client.model, 
                                                  dataloader=trainloader,
                                                  criterion=client.loss, 
                                                  pruning_ratio=max_amount,
                                                  device=self.device)
                    
                    client.mask = snip_pruning(model=client.model, 
                                               dataloader=trainloader,
                                               criterion=client.loss, 
                                               pruning_ratio=max_amount,
                                               device=self.device)
                    
                    local_model = apply_mask(client.model, client.mask)
                    g_model_pruned = apply_mask(g_model_pruned, self.mask)
                    
                    print("SNIP")

            client.set_parameters(local_model)

            # Inicializar localmente com o modelo global
            client.local_initialization(g_model_pruned)

            # Atualizar tempo de envio
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += (time.time() - start_time)
