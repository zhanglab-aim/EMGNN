import torch
import torch.nn as nn
import torch.nn.functional as F
from time import ctime
from torch_geometric.nn import GCNConv,GATConv,GINConv
import numpy as np
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import PairNorm,LayerNorm
 
      
 

class EMGNN(torch.nn.Module):
    def __init__(self, nfeat, hidden_channels, n_layers, nclass, meta_x=None, args=None, data=None, node2idx=None):
        super().__init__()

        self.args = args
        self.linear = nn.Linear(nfeat,hidden_channels)
        self.meta_linear = nn.Linear(nfeat,hidden_channels)
        
        if(args.gcn):
            self.meta_gnn = GCNConv(hidden_channels,hidden_channels)

        elif(args.gat):
            self.meta_gnn = GATConv(hidden_channels,hidden_channels,heads=args.nb_heads,concat=False)
        elif(args.gin):
            self.meta_gnn = GINConv(nn.Sequential(
                                    nn.Linear(hidden_channels, hidden_channels), 
                                    nn.LeakyReLU(), 
                                    nn.BatchNorm1d(hidden_channels),
                                    nn.Linear(hidden_channels, hidden_channels)))

        self.classifier = nn.Linear(hidden_channels,nclass)
        self.dropout = args.dropout
        self.leakyrelu = nn.LeakyReLU(args.alpha)
        self.n_layers = n_layers

        lst = list()
        for i in range(n_layers):
            if(args.gcn):
                lst.append(GCNConv(hidden_channels, hidden_channels))
            elif(args.gat):
                lst.append(GATConv(hidden_channels,hidden_channels,heads=args.nb_heads,concat=False))
            elif(args.gin):
                lst.append(GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LeakyReLU(), nn.BatchNorm1d(hidden_channels),nn.Linear(hidden_channels, hidden_channels),nn.LeakyReLU())))
        self.conv = nn.ModuleList(lst)

        #construct meta graph : meta edge index
        x = data.x.float()
        self.nb_nodes = x.shape[0]
        node_names = np.concatenate(data.node_names,axis=0)
        meta_edge_index = [[],[]]
  
        #node2idx map each node to an idx, same nodes across  graphs are mapped to the same idx, idx in [0,...,Number of different nodes]
        # but we want to create new meta nodes therefore we should increase all idx by the number of all the nodes in the edge index.
        for i,node in enumerate(node_names):
            meta_edge_index[0].append(i)  #input node
            meta_edge_index[1].append(node2idx[tuple(node)]+x.shape[0]) #add metanode       
        
        self.meta_edge_index  = torch.tensor(meta_edge_index).cuda()
        self.meta_edge_index,_ = add_self_loops(self.meta_edge_index)
        #we also have to init some node features for the metanodes and concat them to the x tensor.
        #self.meta_x = torch.zeros((len(node2idx),hidden_channels)).cuda()
        #self.meta_x = torch.rand((len(node2idx),hidden_channels)).cuda()
        self.meta_x = meta_x.cuda()
    
    def forward(self, x, edge_index, data, meta_edge_index = None, explain_x=None, captum=False, explain=False, edge_weight=None):  
        if(captum==True and meta_edge_index!=None):
            meta_x = x[self.nb_nodes:]
            x = x[:self.nb_nodes]
            pass
        if(meta_edge_index!=None):
            self.meta_edge_index = meta_edge_index
     
        #x = torch.concat((x,self.meta_x),dim=0)
        #x = x.cuda()
        #x = data.x.float().cuda()
        number_of_nodes = x.shape[0]
        #edge_index = data.edge_index.cuda()
        ### only for explainer
        if(explain==True):
            
            #x[-1] is the meta node
            meta_x = x[-1].unsqueeze(dim=0)
            x = x[:-1]
            
            x = self.leakyrelu(self.linear(x))

            meta_x = self.leakyrelu(self.meta_linear(meta_x))

            for i in range(1):
                x = self.conv[i](x, edge_index)
                x = self.leakyrelu(x) 
                x = F.dropout(x, self.dropout, training=self.training)
        
            #meta message-passing
            if(self.args.gat):
                #x,attention_scores = self.meta_gnn(torch.concat((x,meta_x),dim=0),meta_edge_index)   #probably this concat slows down the model, maybe we should move it in the init.  
                x = self.meta_gnn(torch.concat((x,meta_x),dim=0),meta_edge_index)   #probably this concat slows down the model, maybe we should move it in the init.  
                #print(attention_scores)
            else:
                x = self.meta_gnn(torch.concat((x,meta_x),dim=0),meta_edge_index)   #probably this concat slows down the model, maybe we should move it in the init.  
            # meta_edge_index: incremented edge index with meta nodes
            x = self.leakyrelu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.classifier(x) #return only predictions for meta nodes
            return F.log_softmax(x,dim=1) 

      
        x = self.leakyrelu(self.linear(x))

        meta_x = self.leakyrelu(self.meta_linear(self.meta_x))

        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.leakyrelu(x) 
            x = F.dropout(x, self.dropout, training=self.training)
        
        #meta message-passing
        if(self.args.gat):
            x = self.meta_gnn(torch.concat((x,meta_x),dim=0),self.meta_edge_index)   #probably this concat slows down the model, maybe we should move it in the init.  
            #print(attention_scores)
        else:
            x = self.meta_gnn(torch.concat((x,meta_x),dim=0),self.meta_edge_index)   #probably this concat slows down the model, maybe we should move it in the init.  
        # meta_edge_index: incremented edge index with meta nodes
        x = self.leakyrelu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x,dim=1) 



class GCN(torch.nn.Module):
    def __init__(self, nfeat, hidden_channels, n_layers, nclass, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(nfeat, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, 1))

        self.convs = torch.nn.ModuleList()
        for layer in range(n_layers):
            self.convs.append(GCNConv(hidden_channels,hidden_channels))


        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index,edge_weight)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
        #return x.softmax(dim=-1) 

 
class MLP(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,nclass,alpha=0.2):
        super().__init__()
        self.alpha = alpha  
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(nfeat,outfeat)
        self.linear_2 = nn.Linear(outfeat,outd_1)
        self.linear_3 = nn.Linear(outd_1,nclass)

    def forward(self,x,edge_index=None,data=None):
        x = self.leakyrelu(self.linear(x))

        x = F.dropout(x,training=self.training)
        
        x = self.leakyrelu(self.linear_2(x))
        x = F.dropout(x, training=self.training)

        
        x= self.linear_3(x)
        return F.log_softmax(x, dim=1)
 

'''
class MultiFrameworkGNN2(torch.nn.Module):
    def __init__(self, nfeat, hidden_channels, n_layers, nclass, n_graphs=None, args=None, number_of_input_nodes=None, data=None, node2idx=None, alpha=0.2, dropout=0.5):
        super().__init__()

        self.args = args
        self.linear = nn.Linear(nfeat,hidden_channels)
        if(args.gcn):
            self.meta_gnn = GCNConv(hidden_channels,hidden_channels)
        elif(args.gat):
            self.meta_gnn = GATConv(hidden_channels,hidden_channels)
        elif(args.gin):
            self.meta_gnn = GINConv(nn.Sequential(
                                    nn.Linear(hidden_channels, hidden_channels), 
                                    nn.LeakyReLU(), 
                                    nn.BatchNorm1d(hidden_channels),
                                    nn.Linear(hidden_channels, hidden_channels)))

        self.classifier = nn.Linear(hidden_channels,2)
        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.n_layers = n_layers
        self.number_of_input_nodes = number_of_input_nodes
        self.node2idx = node2idx   
        self.hidden_channels = hidden_channels

        lst = list()
        for i in range(n_layers):
            if(args.gcn):
                lst.append(GCNConv(hidden_channels,hidden_channels))
            elif(args.gat):
                lst.append(GATConv(hidden_channels,hidden_channels))
            elif(args.gin):
                lst.append(GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LeakyReLU(), nn.BatchNorm1d(hidden_channels),nn.Linear(hidden_channels, hidden_channels),nn.LeakyReLU())))
        self.conv = nn.ModuleList(lst)

        #construct meta graph : meta edge index

        #meta_edge_index = [[],[]]
  
        #node2idx map each node to an idx, same nodes across  graphs are mapped to the same idx, idx in [0,...,Number of different nodes]
        # but we want to create new meta nodes therefore we should increase all idx by the number of all the nodes in the edge index.
        #for i,node in enumerate(data.node_names):
        #    meta_edge_index[0].append(i)  #input node
        #    meta_edge_index[1].append(node2idx[tuple(node)]+x.shape[0]) #add metanode       
        
        #self.meta_edge_index  = torch.tensor(meta_edge_index).cuda()
        #we also have to init some node features for the metanodes and concat them to the x tensor.
        #self.meta_x = torch.zeros((len(node2idx),hidden_channels)).cuda()
        #self.meta_x = torch.rand((len(node2idx),hidden_channels)).cuda()

    def forward(self,x,edge_index,data,edge_weight=None):  

        #node_type = torch.concat([torch.zeros(x.shape[0]-len(self.node2idx)).cuda(),
        #                                torch.ones(len(self.node2idx)).cuda()],dim=0).type(torch.cuda.LongTensor)
        #edge_type = torch.concat([torch.zeros(edge_index.shape[1]-len(data.meta_edge_index)).cuda(),
        #                                torch.ones(len(data.meta_edge_index)).cuda()],dim=0).type(torch.cuda.LongTensor)
        #edge_weight = torch.ones(data.edge_index.shape[1]).unsqueeze(-1).cuda()
        #x = data.x.float().cuda()
        #number_of_nodes = x.shape[0]
        #edge_index = data.edge_index.cuda()
        #node_names = data.node_names
        
        #meta_edge_index = data.meta_edge_index
        #meta_x = data.meta_x
        
        x = self.leakyrelu(self.linear(x))

        #x =  torch.concat([self.leakyrelu(self.linear(x[:self.number_of_input_nodes])),x[self.number_of_input_nodes:]],dim=0)

        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index,edge_weight)
            x = self.leakyrelu(x) 
            x = F.dropout(x, self.dropout, training=self.training)
            
        #meta message-passing
        if(self.args.gat):
            x,attention_scores = self.meta_gnn(x,edge_index,return_attention_weights=True)    
            print(attention_scores)
        else:
            #x_meta_mask = torch.concat([torch.ones((x.shape[0]-len(self.node2idx),self.hidden_channels)).cuda(),
                                        #torch.zeros((len(self.node2idx),self.hidden_channels)).cuda()],dim=0)
            #x = torch.mul(x,x_meta_mask).cuda()
            x = self.meta_gnn(x,edge_index,edge_weight)    

            #x = self.meta_gnn(torch.concat([x,meta_x],dim=0),meta_edge_index)    
        # meta_edge_index: incremented edge index with meta nodes
        x = self.leakyrelu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.classifier(x[self.number_of_input_nodes:])  
        x = self.classifier(x)  
        return F.log_softmax(x,dim=1)
        


'''