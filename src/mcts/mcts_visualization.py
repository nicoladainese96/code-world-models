import numpy as np
import matplotlib.pyplot as plt

from treelib import Node, Tree

def plot_best_value(value_list, ax=None, return_fig=False):
    max_value_list = [max(value_list[:i+1]) for i in range(len(value_list))]

    if ax is None:
        ax = plt.gca()

    ax.plot(value_list, '.', label='Simulation')
    ax.plot(max_value_list, '--', label='Best simulation')
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Code performance")
    ax.legend()
    if return_fig:
        return plt
    else:
        plt.show()
        return None

def print_tree(extra_info, simulator_llm=None, return_tree=False, logger=None, print_non_expanded=False):
    def create_children_nodes(tree, parent_node, parent_name):
        for c in parent_node.children.keys():
            child_name = parent_name+str(c)
            child_node = parent_node.children[c]
            full_child_name = child_name
            if hasattr(child_node, 'node_id') and child_node.node_id is not None:
                full_child_name = f'{child_node.node_id:02d}. {child_name}'
            if child_node.expanded:
                if hasattr(child_node, 'full_code_value'):
                    saved_success_rate = child_node.full_code_value
                    if saved_success_rate is not None:
                        saved_success_rate = f'{saved_success_rate:.2f}'
                    success_rate = f' -- {saved_success_rate}'
                    if simulator_llm is not None:
                        _, prediction_success_rate, _ = simulator_llm.check_sampled_code(child_node.full_code)
                        assert saved_success_rate == prediction_success_rate
                elif simulator_llm is not None:
                    _, prediction_success_rate, _ = simulator_llm.check_sampled_code(child_node.full_code)
                    success_rate = f' -- {prediction_success_rate:.2f}'
                else:
                    success_rate = ''
                terminal = ' - terminal' if child_node.terminal else ''
                bug = ' - bug' if child_node.bug else ''
                tree.create_node(f'{full_child_name}({child_node.visit_count},{child_node.value():.2f}{success_rate}){bug}{terminal}', child_name, parent_name)
                tree = create_children_nodes(tree, child_node, child_name)
            elif print_non_expanded:
                tree.create_node(f'{full_child_name}(unexpanded)', child_name, parent_name)
        return tree

    tree = Tree()
    root = extra_info['root']
    tree.create_node(f'0. ({root.visit_count},{root.value():.2f})', '')
    tree = create_children_nodes(tree, root, '')
    if return_tree:
        return tree
    elif logger is not None:
        logger.info(tree.show(stdout=False))
    else:
        print(tree.show(stdout=False))