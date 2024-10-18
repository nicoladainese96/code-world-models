import torch

from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

class LinearValueModel(torch.nn.Module):
    def __init__(self, w_G_init=0.5, w_L_init=0.5, w_B_init=0.0, logger=DEFAULT_LOGGER):
        super().__init__()
        # Debug / check if it works
        self.w_G = torch.nn.Parameter(torch.tensor(w_G_init))
        self.w_L = torch.nn.Parameter(torch.tensor(w_L_init))
        self.w_B = torch.nn.Parameter(torch.tensor(w_B_init))

        self.logger = logger
        
    def forward(self, v_G, v_L):
        """
        v_G: global value (torch.tensor scalar float)
        v_L: local value (torch.tensor scalar float)
        
        Returns:
        v_pred: predicted value (torch.tensor scalar float)
        """
        # v_G is always present
        # v_L can be None
        if v_L is None:
            v_pred = v_G #+ self.w_B
            return v_pred
        else:
            v_pred = (self.w_G*v_G + self.w_L*v_L)/(self.w_G+self.w_L) #+ self.w_B
            return v_pred
    
class ValuePredictor():
    def __init__(self, c=1, w_G_init=0.5, w_L_init=0.5, w_B_init=0.0, logger=DEFAULT_LOGGER):
        self.value_model = LinearValueModel(w_G_init, w_L_init, w_B_init)
        self.n = 1 + c
        self.lr = 1./self.n
        self.optimizer = torch.optim.SGD(self.value_model.parameters(), lr=self.lr)

        self.logger = logger
        
    def forward(self, v_G, v_L):
        """
        v_G: global value (float)
        v_L: local value (float)
        """
        v_G = torch.tensor(v_G)
        if v_L is not None:
            v_L = torch.tensor(v_L)
        return self.value_model.forward(v_G, v_L)
    
    def update(self, v_G, v_L, v_trg):
        """
        v_G: global value (float)
        v_L: local value (float)
        v_trg: target value (float)
        
        Returns:
        loss: squared difference between prediction and target values (float)
        """
        v_G = torch.tensor(v_G)
        if v_L is not None:
            v_L = torch.tensor(v_L)
        v_trg = torch.tensor(v_trg)
        v_pred = self.value_model.forward(v_G, v_L)
        loss = (v_pred - v_trg)**2
        
        self.logger.debug(f'Weights before update: w_G={float(self.value_model.w_G):.4f} w_L={float(self.value_model.w_L):.4f} w_B={float(self.value_model.w_B):.4f}')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.debug(f'Weights after update: w_G={float(self.value_model.w_G):.4f} w_L={float(self.value_model.w_L):.4f} w_B={float(self.value_model.w_B):.4f}')
        
        self.n += 1
        self.lr = 1./self.n
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

        return v_pred.item(), loss.item() 