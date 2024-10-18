from mloggers import ConsoleLogger, LogLevel
DEFAULT_LOGGER = ConsoleLogger(LogLevel.INFO)

class CodeStateNode():
    def __init__(self, logger=DEFAULT_LOGGER, allow_generate=True, allow_improve=True):
        assert allow_generate or allow_improve, "At least one of the actions should be allowed"
        self.visit_count = 0
        self.value_sum = 0
        self.parent = None
        self.children = {}
        self.reward = 0
        self.expanded = False
        self.terminal = False
        # self.initial_actions = ['g1', 'f1'] # generate and fix
        self.initial_actions = []
        self.initial_actions.append('g1') if allow_generate else None
        self.initial_actions.append('i1') if allow_improve else None

        self.node_code = None
        self.ancestors_code = None
        self.total_code = None # it's not a full program, but just the program so far
        self.full_code = None
        self.full_code_value = None
        self.node_id = None
        
        self.node_length = None
        self.ancestors_length = None
        self.tot_length = None
        self.bug = False

        self.logger = logger
        self.allow_generate = allow_generate
        self.allow_improve = allow_improve
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count 
    
    def expand(
        self, 
        node_code, 
        reward, 
        done, 
        node_length, 
        ancestors_length,
        ancestors_code,
        full_code,
        action_type,
        extra_info=None,
        node_id=None,
        bug_value=0.99
    ):
        self.expanded = True
        self.node_id = node_id
        self.action_type = action_type

        self.logger.info("Terminal node: ", done)
        self.node_code = node_code
        self.ancestors_code = ancestors_code
        self.total_code = ancestors_code+'\n'+node_code
        self.full_code = full_code
        
        self.reward = reward
        self.terminal = done
        self.node_length = node_length
        self.ancestors_length = ancestors_length
        self.tot_length = node_length + ancestors_length
        self.extra_info = extra_info
        
        self.bug = extra_info['bug']
        
        if self.bug and not done and bug_value > 0: # bug_value = 0 means there are no fixes left
            self.initial_actions = ['f1'] # fix only
            self.value_sum += bug_value
        elif self.bug and not done:
            self.initial_actions = []
            
        if not done:
            for action in self.initial_actions:
                self.children[action] = CodeStateNode(logger=self.logger, allow_generate=self.allow_generate, allow_improve=self.allow_improve)
                self.children[action].action = action