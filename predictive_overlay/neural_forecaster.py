from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QuantumForecaster:
    def __init__(self, num_qubits=4):
        self.qc = QuantumCircuit(num_qubits)
        self._build_entanglement_layer()
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=["ZZII", "IIZZ"]
        )
      
    def _build_entanglement_layer(self):
        """Creates quantum feature map"""
        self.qc.h(range(self.qc.num_qubits))
        self.qc.cz(0, 1)
        self.qc.cz(2, 3)
          
    def predict(self, market_data):
        return self.qnn.forward(market_data)
