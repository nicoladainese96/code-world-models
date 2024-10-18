# Needed to access CodeStateNode class when loading tree
from gif_mcts import CodeStateNode
import os
import sys
import numpy as np

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

# Import src code from parent directory
#PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath('')))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.code_helpers import load_tree

def get_action_type(node):
    for i in range(len(node) - 1, -1, -1):
        if node[i].isalpha():
            return node[i]

def get_node_color(node):
    if get_action_type(node) == 'g':
        return 'lightgreen'
    elif get_action_type(node) == 'f':
        return 'lightcoral'
    elif get_action_type(node) == 'i':
        return 'lightblue'
    else:
        return 'lightgray'

def get_traces(tree):
    edge_x = []
    edge_y = []
    for edge in tree.edges():
        x0, y0 = tree.nodes[edge[0]]['pos']
        x1, y1 = tree.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in tree.nodes():
        x, y = tree.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"{node}{(' - '+tree.nodes[node]['props']['full_code_value']) if tree.nodes[node]['props']['full_code_value'] is not None else ''}" for node in tree.nodes()],
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            color=[get_node_color(node) for node in tree.nodes()],
            size=25,
        ))

    return edge_trace, node_trace

def get_node_props(node):
    props = {
        "visit_count": node.visit_count,
        "value_sum": node.value_sum,
        "reward": node.reward,
        "node_code": node.node_code,
        "ancesors_code": node.ancestors_code,
        "total_code": node.total_code,
        "full_code": node.full_code,
        "bug" : node.bug,
    }
    if hasattr(node, 'full_code_value') and node.full_code_value is not None:
        value = float(node.full_code_value)
        saved_success_rate = str(np.round(value, 2))
    else:
        saved_success_rate = None
    props['full_code_value'] = saved_success_rate
    
    extra_info = node.extra_info
    props['critique'] = extra_info.get('critique', None)
    
    return props

def build_tree(extra_info):
    tree = nx.DiGraph()
    tree.add_node('root')
    props = {'root': get_node_props(extra_info['root'])}

    def create_children_nodes(tree, parent_node, parent_name):
        for c in parent_node.children.keys():
            child_name = parent_name + \
                str(c) if parent_name != 'root' else str(c)
            child_node = parent_node.children[c]
            full_child_name = child_name
            if hasattr(child_node, 'node_id') and child_node.node_id is not None:
                full_child_name = f'{child_node.node_id:02d}. {child_name}'
            if child_node.expanded:
                tree.add_node(f'{full_child_name}')
                tree.add_edge(parent_name, full_child_name)
                props[full_child_name] = get_node_props(child_node)
                create_children_nodes(tree, child_node, full_child_name)
        return

    create_children_nodes(tree, extra_info['root'], 'root')
    pos = graphviz_layout(tree, prog='dot')
    for node in tree.nodes:
        tree.nodes[node]['pos'] = pos[node]
        tree.nodes[node]['props'] = props[node]
    return tree

def main():
    app = Dash(__name__, external_scripts=[{"src": "https://cdn.tailwindcss.com"}])
    
    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='graph',
                className='h-[95vh]',
                figure=go.Figure(data=[],
                                layout=go.Layout(
                                    title='MCTS Tree Visualizer',
                                    titlefont_size=24,
                                    showlegend=False,
                                    hovermode='closest',
                                    autosize=True,
                                    height=None,
                                    annotations=[dict(
                                        text="",
                                        showarrow=False,
                                        xref="paper", yref="paper",
                                        x=0.005, y=-0.002)],
                                    xaxis=dict(
                                        showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(
                                        showgrid=False, zeroline=False, showticklabels=False),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                ))
            ),
            html.Div([
                html.Button("Load Tree", id="load-tree", className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 mb-8 rounded-full w-36"),
            ], className="flex justify-center"),
        ], className='basis-3/5 h-screen bg-gray-100 flex flex-col'),
        dcc.Store(id='node-data'),
        dcc.Markdown(id='node-info', className='basis-2/5 h-screen bg-gray-200 overflow-y-auto overflow-x-auto border-none p-4')
    ], className='flex flex-row justify-between h-screen w-screen bg-gray-100')
    
    @app.callback(
        Output('node-data', 'data'),
        Input('graph', 'clickData'),
    )
    def update_node_data(clickData):
        if clickData is None:
            return None
        return clickData['points'][0]['text']  # Store the clicked node

    @app.callback(
        Output('node-info', 'children'),
        Input('node-data', 'data')
    )
    def update_node_info(node):
        if node is None:
            return ''
        node = node.split('-')[0].strip()
        print(f"Clicked node: {node}")
        info = tree.nodes[node]['props']
        display = f"""# Node Info - {node}\n\n- Visit Count: {info['visit_count']}\n- Value Sum: {info['value_sum']}\n- Reward: {info['reward']}\n- Bug: {info['bug']} \
            \n- Full Code Value: {info['full_code_value']}\n\n """
        if info['critique'] is not None:
            display += f"## Critique: \n{info['critique']}\n\n"
        display += f"""\n\n## Code\n```python\n{info['full_code'].replace("`", "")}\n```\n"""
        return display
    
    @app.callback(
        Output('graph', 'figure'),
        Input('load-tree', 'n_clicks')
    )
    def load_tree_folder(n_clicks):
        # Event handler for some reason gets called on page load (seems like a known issue in Dash) so skipping the first call
        if n_clicks is None or n_clicks == 0:
            figureData = []
        else:
            app = QApplication([])
            options = QFileDialog.Options()
            # options |= QFileDialog.DontUseNativeDialog
            tree_dir = QFileDialog.getExistingDirectory(None, "Select Tree Directory", os.path.join(PROJECT_ROOT, "results"), options=options)
            print(f"Selected tree directory: {tree_dir}")
            
            try:
                loaded_root, loaded_mcts_state = load_tree(tree_dir)
                global tree
                tree = build_tree(extra_info=loaded_mcts_state['extra_info'])
            except Exception as e:
                error_dialog = QMessageBox()
                error_dialog.setWindowTitle("Error")
                error_dialog.setText(f"Error loading tree\n{e}")
                error_dialog.exec_()
                figureData = []
        
            edge_trace, node_trace = get_traces(tree)
            figureData = [edge_trace, node_trace]
            print(f"Loaded tree with {len(tree.nodes)} nodes and {len(tree.edges)} edges")
        
        return go.Figure(data=figureData,
                            layout=go.Layout(
                                title='MCTS Tree Visualizer',
                                titlefont_size=24,
                                showlegend=False,
                                hovermode='closest',
                                autosize=True,
                                height=None,
                                annotations=[dict(
                                    text="",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002)],
                                xaxis=dict(
                                    showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(
                                    showgrid=False, zeroline=False, showticklabels=False),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                            ))
    
    app.run_server(debug=False)
    

if __name__ == '__main__':
    main()
