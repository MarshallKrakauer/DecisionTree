"""Functions that can be used to view the decision tree splits"""

def print_current_level(node, level):
    """
    Prints all nodes at the given level of the tree

    :param node: root node from which to check for values
    :param level: depth of tree (starting at 0) from which to print
    """
    if level < 0:
        raise ValueError("minimum depth is 0")

    if node is None:
        return
    if level == 0:
        print(node)
    elif level > 0:
        print_current_level(node.left_child, level-1)
        print_current_level(node.right_child, level-1)

def print_breadth_first(node):
    """
    Print the nodes depth first (starting from 0 and going to the bottom).

    :param node: root_node of decision tree
    """
    for i in range(0, node.max_depth+1):
        print_current_level(node,i)