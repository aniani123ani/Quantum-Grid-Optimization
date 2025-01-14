import numpy as np
from qiskit import Aer, QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import algorithm_globals
import matplotlib.pyplot as plt

def generate_qubo(num_jobs, num_nodes):
    """Generate a random QUBO matrix for job scheduling."""
    np.random.seed(42)
    return np.random.randint(-10, 10, size=(num_jobs * num_nodes, num_jobs * num_nodes))

num_jobs = 3
num_nodes = 3
qubo_matrix = generate_qubo(num_jobs, num_nodes)

num_qubits = len(qubo_matrix)
circuit = QuantumCircuit(num_qubits)
params = [Parameter(f'theta_{i}') for i in range(num_qubits)]


for i in range(num_qubits):
    circuit.ry(params[i], i)
    if i < num_qubits - 1:
        circuit.cx(i, i + 1)

def evaluate_cost(qubo, bitstring):
    """Compute the cost of a bitstring based on the QUBO matrix."""
    state = np.array([int(bit) for bit in bitstring])
    return state @ qubo @ state.T

def objective_function(theta_values):
    """Objective function for the optimizer."""
    bound_circuit = circuit.bind_parameters({param: theta for param, theta in zip(params, theta_values)})
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(bound_circuit, simulator).result()
    statevector = result.get_statevector()
    probabilities = np.abs(statevector) ** 2
    expected_cost = 0

    for i, prob in enumerate(probabilities):
        bitstring = f"{i:0{num_qubits}b}"
        expected_cost += prob * evaluate_cost(qubo_matrix, bitstring)

    return expected_cost

optimizer = COBYLA(maxiter=100)
algorithm_globals.random_seed = 42
initial_theta = np.random.random(len(params)) * 2 * np.pi

costs = []

def callback(theta_values):
    costs.append(objective_function(theta_values))

result = optimizer.minimize(fun=objective_function, x0=initial_theta, callback=callback)

print("Optimal Parameters:", result.x)
print("Minimum Cost Achieved:", result.fun)

plt.figure(figsize=(10, 6))
plt.plot(costs, marker='o')
plt.title('Optimization Convergence')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid()
plt.show()

def random_search(qubo, num_samples=1000):
    """Perform a classical random search for comparison."""
    best_cost = float('inf')
    for _ in range(num_samples):
        bitstring = np.random.choice([0, 1], size=num_qubits)
        cost = evaluate_cost(qubo, ''.join(map(str, bitstring)))
        if cost < best_cost:
            best_cost = cost
    return best_cost

classical_best = random_search(qubo_matrix)
print("Classical Best Cost:", classical_best)
print("Quantum-Optimized Cost:", result.fun)

def test_with_realistic_data():
    """Test the optimization algorithm with a realistic dataset."""
    realistic_qubo = np.array([
        [-1, 2, -3],
        [2, -4, 5],
        [-3, 5, -6]
    ])
    print("Testing with Realistic QUBO Matrix:")
    print(realistic_qubo)

    def realistic_objective_function(theta_values):
        bound_circuit = circuit.bind_parameters({param: theta for param, theta in zip(params, theta_values)})
        simulator = Aer.get_backend('statevector_simulator')
        result = execute(bound_circuit, simulator).result()
        statevector = result.get_statevector()
        probabilities = np.abs(statevector) ** 2
        expected_cost = 0

        for i, prob in enumerate(probabilities):
            bitstring = f"{i:0{num_qubits}b}"
            expected_cost += prob * evaluate_cost(realistic_qubo, bitstring)

        return expected_cost

    realistic_optimizer = COBYLA(maxiter=100)
    realistic_costs = []

    def realistic_callback(theta_values):
        realistic_costs.append(realistic_objective_function(theta_values))

    result = realistic_optimizer.minimize(fun=realistic_objective_function, x0=initial_theta, callback=realistic_callback)

    print("Realistic Test Optimal Parameters:", result.x)
    print("Realistic Test Minimum Cost Achieved:", result.fun)

    plt.figure(figsize=(10, 6))
    plt.plot(realistic_costs, marker='o', color='orange')
    plt.title('Optimization Convergence on Realistic Data')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid()
    plt.show()

test_with_realistic_data()

print("\nExtensibility:")
print("This code is designed to scale for larger problems by modifying the QUBO generation and increasing the qubit count.")
print("Integration with distributed systems can be achieved by replacing the random QUBO generation with real-world data feeds.")
#Have fun guys!! Thanks!
