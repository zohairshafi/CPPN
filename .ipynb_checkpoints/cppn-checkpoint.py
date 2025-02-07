from utils import *
from torch_geometric.datasets import ZINC

class CPPN(torch.nn.Module):
    
    def __init__(self, in_feats, num_classes = None, num_layers = 2, num_hidden = 128, dropout = 0.1):
        super(CPPN, self).__init__()
        
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.in_feats, self.num_hidden))
        # self.class_layer_zero = torch.nn.Linear(self.num_hidden, self.num_hidden)
        # self.class_layer = torch.nn.Linear(self.num_hidden, self.num_classes)
        
        for _ in range(self.num_layers - 2):
            self.lins.append(torch.nn.Linear(self.num_hidden, self.num_hidden))
            
        self.lins.append(torch.nn.Linear(self.num_hidden, 1))
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
            
    def forward(self, x):
        
        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = F.relu(x)

        # c = self.class_layer_zero(x)
        # c = self.class_layer(c)
        
        x = self.lins[-1](x)
        
        return x #torch.sigmoid(x)

z = ZINC(root = './data', subset = True)

mol_loss = []

cppn = CPPN(in_feats = 61, #num_feats + context.shape[0],
                num_layers = 6,
                num_classes = 0,
                num_hidden = 512,
                dropout = 0.3).to(device)
    
optimizer = optim.Adam(cppn.parameters(),
                       lr = 1e-4,
                       weight_decay = 1e-7)
cppn.train()

for z_idx in range(10000):
    
    edge_list = z[z_idx].edge_index.detach().cpu().numpy()
    graph = nx.from_edgelist([(edge_list[0][idx], edge_list[1][idx]) for idx in range(edge_list.shape[1])])
    adj = torch.Tensor(nx.to_numpy_array(graph)).to(device)

    # Curvature
    deg = torch.sum(adj, axis = 1)
    curv = 4 - deg.unsqueeze(1) - deg.unsqueeze(0)

    # Hop Distance 
    len_adj = []
    for n in tqdm(graph.nodes):
        lengths = dict(nx.single_source_dijkstra_path_length(graph, n))
        len_adj.append(np.array(list(dict(sorted(lengths.items(), key = lambda item: item[0])).values())))
    len_adj = np.array(len_adj)
    
    
   
    f = z[z_idx].x.detach().cpu().flatten().numpy()
    f = (f - np.min(f)) / np.ptp(f)


    node_features = F.one_hot(z[z_idx].x, num_classes = 21).detach().cpu().numpy()
    edge_features = F.one_hot(z[z_idx].edge_attr, num_classes = 4).detach().cpu().numpy()

    # Coordinates
    x_mat = np.tile((np.arange(adj.shape[0]) / adj.shape[0]), (adj.shape[0], 1)).T 
    y_mat = np.tile((np.arange(adj.shape[1]) / adj.shape[1]), (adj.shape[1], 1)).T 
    
    # Node Features
    # np_feat = (features @ features.T).detach().cpu().numpy()
    # np_feat = (np_feat - np.min(np_feat)) / np.ptp(np_feat)
    node_feat_map = np.zeros((adj.shape[0], adj.shape[1], 2 * node_features.shape[-1]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            node_feat_map[i][j] = np.hstack([node_features[0], node_features[1]])
    
    edge_feat_map = np.zeros((adj.shape[0], adj.shape[1], edge_features.shape[-1]))
    for i in range(edge_list.shape[1]):
        edge_feat_map[edge_list[0][i]][edge_list[1][i]] = edge_features[i]
            
    
    # Hop Length 
    norm_len_adj = (len_adj - np.min(len_adj)) / np.ptp(len_adj)
    
    # Input Matrix
    in_mat = np.stack([x_mat, y_mat, norm_len_adj],
                      -1)
   
    
    # Structural Features 
    sense_feat_dict, sense_features = get_sense_features(graph)
    feat_map = np.zeros((sense_features.shape[0], sense_features.shape[0], 2 * sense_features.shape[1]))
    for i in range(sense_features.shape[0]):
        for j in range(sense_features.shape[0]):
            feat_map[i, j, :] = np.concatenate([sense_features[i], sense_features[j]])
 
    in_mat = np.concatenate([in_mat, feat_map, edge_feat_map, node_feat_map], axis = -1)
    
    in_mat = torch.Tensor(in_mat)
    in_mat = in_mat.to(device)
    
    
    num_feats = 3 + feat_map.shape[-1] + edge_feat_map.shape[-1] + node_feat_map.shape[-1]
    
    for e in range(2000):
        optimizer.zero_grad()
        out = cppn(in_mat)
        out = out.reshape(adj.shape[0], adj.shape[0])
        out = (out + out.T) / 2
        recon_adj = out 
        recon_deg = torch.sum(recon_adj, axis = 1)
        recon_curv = 4 - recon_deg.unsqueeze(1) - recon_deg.unsqueeze(0)

        curv_loss = torch.sum(torch.square(recon_curv - curv))
        
        # out_list.append(out.detach().cpu().numpy())
        diff = out - adj
        #diff = diff[r, c]
        edge_loss = torch.sum(torch.square(diff)) 
        loss = edge_loss + (1e-2 * curv_loss) 
        loss.backward()
        optimizer.step()
        # print ("Epoch : ", e, "Loss : ", loss.item(), end = '\r') #, "(Class Loss : ", class_loss.item(), ")", end = '\r')
    print ("Molecule ", z_idx, " | Loss : ", loss.item())
    mol_loss.append(loss.item())

    if z_idx % 10 == 0:
        torch.save(cppn.state_dict(), './cppn.pkl')



















