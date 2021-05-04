def walk(obj):

    def set_node_by_path(obj, path, val):
        temp = obj
        #iterate through this level of keys
        for k in temp.keys() if type(temp) == dict else range(len(temp)):
            #if we're on the right track
            if k == path[0]:
                #Are we at the end?
                if len(path) == 1:
                    temp[k] = val
                else:
                    #recurse against temp[k]
                    temp[k] = set_node_by_path(temp[k], path[1::], val)
        return temp


    def isiterable(o):
        return getattr(obj, '__iter__', False)

    def _walk(sub_obj, cb, acc=[]):
        # It would be nice to have this actually output a full object instead of
        # just attaching things to it later.
        class Node(object):
            def __init__(self, node, acc):
                self.value = node
                self.key = acc[-1]
                self.is_leaf = isiterable(node)
                self.is_root = (acc == [])
                self.path = acc
                self.level = len(acc)
            def set(self, v):
                return set_node_by_path(obj, self.path, v)


        # either the keys of the dict, or the indices of the list/tuple
        if getattr(sub_obj, '__iter__', False):
            for k in sub_obj.keys() if type(sub_obj) == dict else range(len(sub_obj)):
                #print obj[k]
                node = Node(sub_obj[k], acc+[k])
                #print(node)
                cb(node)
                if isiterable(sub_obj):
                    _walk(sub_obj[k], cb, acc+[k])

    return lambda cb: _walk(obj, cb)

if __name__=="__main__":
    tree = {'a': 1, 'b': [9, 9, 6, 7, 6]}

    print("Tree before: "+repr(tree))

    @walk(tree)
    def node_printer(node):
        if node.value == [9, 9, 6, 7, 6] :
            node.set([9, 9, 7, 7, 5])

    print("Tree after: "+repr(tree))