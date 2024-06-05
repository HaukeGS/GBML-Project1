import pytest
import sys
from os.path import dirname, abspath

# Add the parent directory to the system path
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from main import find_relationships, graph_is_DAG

nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]

edges = [
    ("A", "B"), ("A", "C"), 
    ("B", "D"), ("B", "E"), 
    ("C", "F"), 
    ("D", "G"), ("D", "H"), 
    ("E", "I"), ("E", "J"), 
    ("F", "K"), 
    ("K", "L"), ("K", "M"), ("K", "N")
]

test_cases = [
    ("O", [],[],["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])
    # More test cases should be added to cover different scenarios
]

@pytest.mark.parametrize("node, expected_parents, expected_descendants, expected_non_descendants", test_cases)
def test_find_relationships_0(node, expected_parents, expected_descendants, expected_non_descendants):
    parents, descendants, non_descendants = find_relationships(node,nodes,edges)
    assert sorted(parents) == sorted(expected_parents)
    assert sorted(descendants) == sorted(expected_descendants)
    assert sorted(non_descendants) == sorted(expected_non_descendants)

test_cases = [
    ("A", [], ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"], ["O"]),
]

@pytest.mark.parametrize("node, expected_parents, expected_descendants, expected_non_descendants", test_cases)
def test_find_relationships_1(node, expected_parents, expected_descendants, expected_non_descendants):
    parents, descendants, non_descendants = find_relationships(node,nodes,edges)
    assert sorted(parents) == sorted(expected_parents)
    assert sorted(descendants) == sorted(expected_descendants)
    assert sorted(non_descendants) == sorted(expected_non_descendants)

test_cases = [
    ("B", ["A"], ["D", "E", "G", "H", "I", "J"], ["A", "C", "F", "K", "L", "M", "N", "O"]),
]

@pytest.mark.parametrize("node, expected_parents, expected_descendants, expected_non_descendants", test_cases)
def test_find_relationships_2(node, expected_parents, expected_descendants, expected_non_descendants):
    parents, descendants, non_descendants = find_relationships(node,nodes,edges)
    assert sorted(parents) == sorted(expected_parents)
    assert sorted(descendants) == sorted(expected_descendants)
    assert sorted(non_descendants) == sorted(expected_non_descendants)

test_cases = [
    {
        "edges": [
            ("A", "B"), ("B", "C"), ("C", "A"),
            ("C", "D"), ("D", "E"), ("E", "C"),
            ("F", "F"),
            ("G", "H"), ("H", "I")
        ],
        "expected_is_dag": False
    },
    {
        "edges": [
            ("A", "B"), ("A", "C"), ("B", "D"),
            ("C", "D"), ("C", "E"), ("D", "F"),
            ("E", "F"), ("F", "G")
        ],
        "expected_is_dag": True
    },
    {
        "edges": [
            ("A", "B"), ("B", "A"),
            ("C", "D"), ("D", "E"), ("E", "C"),
            ("F", "F"), ("G", "G")
        ],
        "expected_is_dag": False
    },
    {
        "edges": [
            ("A", "B"), ("B", "C"), ("C", "D"),
            ("D", "E"), ("E", "F"), ("F", "G"),
            ("G", "H"), ("H", "A"),
            ("I", "J"), ("J", "K")
        ],
        "expected_is_dag": False
    },
]

@pytest.mark.parametrize("test_data", [test_cases[0]])
def test_graph_is_DAG_0(test_data):
    edges = test_data["edges"]
    expected_is_dag = test_data["expected_is_dag"]
    actual_is_dag = graph_is_DAG(edges)
    assert actual_is_dag == expected_is_dag

@pytest.mark.parametrize("test_data", [test_cases[1]])
def test_graph_is_DAG_1(test_data):
    edges = test_data["edges"]
    expected_is_dag = test_data["expected_is_dag"]
    actual_is_dag = graph_is_DAG(edges)
    assert actual_is_dag == expected_is_dag

@pytest.mark.parametrize("test_data", [test_cases[2]])
def test_graph_is_DAG_2(test_data):
    edges = test_data["edges"]
    expected_is_dag = test_data["expected_is_dag"]
    actual_is_dag = graph_is_DAG(edges)
    assert actual_is_dag == expected_is_dag

@pytest.mark.parametrize("test_data", [test_cases[3]])
def test_graph_is_DAG_3(test_data):
    edges = test_data["edges"]
    expected_is_dag = test_data["expected_is_dag"]
    actual_is_dag = graph_is_DAG(edges)
    assert actual_is_dag == expected_is_dag

if __name__ == "__main__":
    pytest.main()
