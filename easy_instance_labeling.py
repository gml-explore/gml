class EasyInstanceLabeling:
    '''
    用于简单实例标注的类，提供了一些用于标记easy的方法
    return easys
    '''
    def __init__(self,variables,features):
        self.variables = variables
        self.features  = features

    def label_easy_by_file(easys):
        '''根据提供的easy列表标出variables中的Easy'''
        easy_keys = ['var_id', 'label']
        if easys != None and type(easys) == list and len(easys) >= 1:
          #check data
            for attribute in easy_keys:
               for easy in easys:
                   if attribute not in easy:
                       raise ValueError('easys has no key: '+attribute)
            #init all variables easy attribute
            for var in self.variables:
                var['is_easy'] = False
                var['is_evidence'] = False
            for easy in easys:
                var_index = easy['var_id']
                self.variables[var_index]['is_easy'] = True
                self.variables[var_index]['is_evidence'] = True
                self.variables[var_index]['label'] = easy['label']
