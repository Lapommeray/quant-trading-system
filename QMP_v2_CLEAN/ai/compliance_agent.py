"""
Compliance Agent Module

This module implements the Compliance Agent that ensures all trading activities
comply with regulations using a knowledge graph approach.
"""

import os
import json
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RegulatoryKnowledgeGraph:
    """
    Regulatory Knowledge Graph for storing and querying regulatory information.
    
    This class simulates a vector database for regulatory knowledge using embeddings
    and similarity search.
    """
    
    def __init__(self, embedding_model=None):
        """Initialize the Regulatory Knowledge Graph"""
        self.embedding_model = embedding_model
        self.regulations = []
        self.regulation_embeddings = []
        
        self._load_regulatory_data()
        
        print("Regulatory Knowledge Graph initialized with regulatory data")
    
    def _load_regulatory_data(self):
        """Load regulatory data from files or initialize with default data"""
        default_regulations = [
            {
                "id": "SEC_15c3-5",
                "name": "SEC Rule 15c3-5",
                "description": "Market Access Rule requiring risk management controls",
                "requirements": [
                    "Pre-trade risk controls",
                    "Credit limits",
                    "Erroneous order controls",
                    "Post-trade surveillance"
                ],
                "asset_classes": ["equities", "options", "futures", "forex", "crypto"],
                "jurisdictions": ["US"],
                "penalties": "Up to $1M per violation"
            },
            {
                "id": "MiFID_II",
                "name": "MiFID II",
                "description": "Markets in Financial Instruments Directive II",
                "requirements": [
                    "Best execution",
                    "Transaction reporting",
                    "Position limits",
                    "Algorithmic trading controls"
                ],
                "asset_classes": ["equities", "bonds", "derivatives", "forex"],
                "jurisdictions": ["EU"],
                "penalties": "Up to 5M EUR or 10% of annual turnover"
            },
            {
                "id": "FATF_TR",
                "name": "FATF Travel Rule",
                "description": "Financial Action Task Force rule for crypto transfers",
                "requirements": [
                    "KYC for transfers",
                    "Beneficiary information",
                    "Originator information",
                    "Record keeping"
                ],
                "asset_classes": ["crypto"],
                "jurisdictions": ["Global"],
                "penalties": "Varies by jurisdiction"
            },
            {
                "id": "CFTC_REG_AT",
                "name": "CFTC Regulation AT",
                "description": "Regulation Automated Trading",
                "requirements": [
                    "Risk controls",
                    "Development and testing",
                    "Source code maintenance",
                    "Annual certification"
                ],
                "asset_classes": ["futures", "commodities", "swaps"],
                "jurisdictions": ["US"],
                "penalties": "Up to $1M per day per violation"
            },
            {
                "id": "MAR",
                "name": "Market Abuse Regulation",
                "description": "EU regulation on market abuse",
                "requirements": [
                    "Prohibition of insider dealing",
                    "Prohibition of market manipulation",
                    "Disclosure requirements",
                    "Suspicious transaction reporting"
                ],
                "asset_classes": ["equities", "bonds", "derivatives", "commodities"],
                "jurisdictions": ["EU"],
                "penalties": "Up to 15M EUR or 15% of annual turnover"
            }
        ]
        
        for regulation in default_regulations:
            self.add_regulation(regulation)
    
    def add_regulation(self, regulation):
        """
        Add a regulation to the knowledge graph
        
        Parameters:
        - regulation: Dict containing regulation data
        
        Returns:
        - True if added successfully, False otherwise
        """
        if not isinstance(regulation, dict):
            return False
        
        text_repr = f"{regulation['name']}: {regulation['description']}. "
        text_repr += f"Requirements: {', '.join(regulation['requirements'])}. "
        text_repr += f"Asset classes: {', '.join(regulation['asset_classes'])}. "
        text_repr += f"Jurisdictions: {', '.join(regulation['jurisdictions'])}."
        
        embedding = self._generate_embedding(text_repr)
        
        if embedding is not None:
            self.regulations.append(regulation)
            self.regulation_embeddings.append(embedding)
            return True
        
        return False
    
    def _generate_embedding(self, text):
        """
        Generate embedding for text
        
        Parameters:
        - text: Text to generate embedding for
        
        Returns:
        - Embedding vector or None if embedding model is not available
        """
        if self.embedding_model is None:
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val)
            return np.random.rand(768)  # Simulate a 768-dim embedding
        
        try:
            return self.embedding_model.encode([text])[0]
        except AttributeError:
            try:
                return self.embedding_model.transform([text]).toarray()[0]
            except:
                return None
    
    def similarity_search(self, query, k=3):
        """
        Search for regulations similar to the query
        
        Parameters:
        - query: Query text
        - k: Number of results to return
        
        Returns:
        - List of regulations sorted by similarity
        """
        if not self.regulations:
            return []
        
        query_embedding = self._generate_embedding(query)
        
        if query_embedding is None:
            return []
        
        similarities = cosine_similarity([query_embedding], self.regulation_embeddings)[0]
        
        sorted_indices = np.argsort(similarities)[::-1][:k]
        
        return [self.regulations[i] for i in sorted_indices]

class ComplianceAgent:
    """
    Compliance Agent for ensuring regulatory compliance of trading activities.
    
    This class uses a Regulatory Knowledge Graph to check orders and trading activities
    against regulatory requirements.
    """
    
    def __init__(self, embedding_model=None):
        """Initialize the Compliance Agent"""
        self.regulatory_knowledge_graph = RegulatoryKnowledgeGraph(embedding_model)
        self.compliance_log = []
        self.violation_count = 0
        self.last_check_time = None
        
        print("Compliance Agent initialized with Regulatory Knowledge Graph")
    
    def check_order(self, order):
        """
        Check if an order complies with regulations
        
        Parameters:
        - order: Dict containing order data
        
        Returns:
        - Dict with compliance check results
        """
        if not isinstance(order, dict):
            return {"compliant": False, "reason": "Invalid order format"}
        
        required_fields = ["symbol", "side", "quantity", "price"]
        for field in required_fields:
            if field not in order:
                return {"compliant": False, "reason": f"Missing required field: {field}"}
        
        query = f"Regulations for {order['symbol']} {order['side']} order"
        regulations = self.regulatory_knowledge_graph.similarity_search(query, k=3)
        
        compliance_issues = []
        for regulation in regulations:
            issues = self._check_regulation_compliance(order, regulation)
            if issues:
                compliance_issues.extend(issues)
        
        self._log_compliance_check(order, regulations, compliance_issues)
        
        if compliance_issues:
            self.violation_count += 1
            return {
                "compliant": False,
                "reason": "; ".join(compliance_issues),
                "regulations": [r["name"] for r in regulations]
            }
        
        return {
            "compliant": True,
            "regulations": [r["name"] for r in regulations]
        }
    
    def _check_regulation_compliance(self, order, regulation):
        """
        Check if an order complies with a specific regulation
        
        Parameters:
        - order: Dict containing order data
        - regulation: Dict containing regulation data
        
        Returns:
        - List of compliance issues
        """
        issues = []
        
        asset_class = self._determine_asset_class(order["symbol"])
        if asset_class not in regulation["asset_classes"]:
            return []  # Regulation doesn't apply to this asset class
        
        if regulation["id"] == "SEC_15c3-5":
            if order["quantity"] > 1000000:
                issues.append("Potential erroneous order: Quantity exceeds 1,000,000")
            
            if "market_price" in order and order["price"] > 0:
                price_diff = abs(order["price"] - order["market_price"]) / order["market_price"]
                if price_diff > 0.5:
                    issues.append(f"Price deviates from market by {price_diff:.2%}")
        
        elif regulation["id"] == "MiFID_II":
            if "market_price" in order and order["price"] > 0:
                if order["side"] == "BUY" and order["price"] > order["market_price"] * 1.05:
                    issues.append("Potential best execution violation: Buy price 5% above market")
                elif order["side"] == "SELL" and order["price"] < order["market_price"] * 0.95:
                    issues.append("Potential best execution violation: Sell price 5% below market")
        
        elif regulation["id"] == "MAR":
            if "time" in order and "previous_orders" in order:
                if self._detect_layering_pattern(order, order["previous_orders"]):
                    issues.append("Potential layering/spoofing pattern detected")
        
        return issues
    
    def _determine_asset_class(self, symbol):
        """
        Determine the asset class of a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Asset class string
        """
        if symbol.endswith("USD") or symbol.endswith("USDT") or symbol.endswith("BTC"):
            return "crypto"
        elif symbol in ["ES", "NQ", "CL", "GC", "SI"]:
            return "futures"
        elif symbol.startswith("^"):
            return "indices"
        elif len(symbol) <= 5:
            return "equities"
        else:
            return "unknown"
    
    def _detect_layering_pattern(self, current_order, previous_orders):
        """
        Detect potential layering/spoofing patterns
        
        Parameters:
        - current_order: Current order
        - previous_orders: List of previous orders
        
        Returns:
        - True if pattern detected, False otherwise
        """
        if len(previous_orders) < 5:
            return False
        
        buy_count = 0
        sell_count = 0
        cancel_count = 0
        
        current_time = current_order.get("time", datetime.datetime.now())
        
        for order in previous_orders:
            order_time = order.get("time", current_time)
            time_diff = (current_time - order_time).total_seconds()
            
            if time_diff <= 300:  # Within 5 minutes
                if order.get("side") == "BUY":
                    buy_count += 1
                elif order.get("side") == "SELL":
                    sell_count += 1
                
                if order.get("status") == "CANCELED":
                    cancel_count += 1
        
        if (buy_count > 10 and sell_count <= 2) or (sell_count > 10 and buy_count <= 2):
            if cancel_count >= 8:
                return True
        
        return False
    
    def _log_compliance_check(self, order, regulations, issues):
        """
        Log a compliance check
        
        Parameters:
        - order: Dict containing order data
        - regulations: List of regulations checked
        - issues: List of compliance issues
        """
        self.last_check_time = datetime.datetime.now()
        
        log_entry = {
            "timestamp": self.last_check_time.isoformat(),
            "order": {
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "price": order["price"]
            },
            "regulations_checked": [r["name"] for r in regulations],
            "issues": issues,
            "compliant": len(issues) == 0
        }
        
        self.compliance_log.append(log_entry)
        
        if len(self.compliance_log) > 1000:
            self.compliance_log = self.compliance_log[-1000:]
    
    def get_compliance_statistics(self):
        """
        Get statistics about compliance checks
        
        Returns:
        - Dict with compliance statistics
        """
        if not self.compliance_log:
            return {
                "checks_performed": 0,
                "violation_rate": 0.0,
                "last_check_time": None
            }
        
        violation_count = sum(1 for entry in self.compliance_log if not entry["compliant"])
        
        return {
            "checks_performed": len(self.compliance_log),
            "violation_count": violation_count,
            "violation_rate": violation_count / len(self.compliance_log),
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None
        }
    
    def explain_decision(self, order_id):
        """
        Generate an explanation for a compliance decision
        
        Parameters:
        - order_id: ID of the order to explain
        
        Returns:
        - Explanation string
        """
        
        for entry in reversed(self.compliance_log):
            if entry["order"].get("id") == order_id:
                if entry["compliant"]:
                    return f"Order {order_id} was compliant with all applicable regulations: {', '.join(entry['regulations_checked'])}"
                else:
                    return f"Order {order_id} violated regulations due to: {'; '.join(entry['issues'])}"
        
        return f"No compliance record found for order {order_id}"
    
    def generate_compliance_report(self, start_time=None, end_time=None):
        """
        Generate a compliance report for a time period
        
        Parameters:
        - start_time: Start time for the report (datetime)
        - end_time: End time for the report (datetime)
        
        Returns:
        - Dict with compliance report data
        """
        if not self.compliance_log:
            return {"error": "No compliance data available"}
        
        if start_time is None:
            start_time = datetime.datetime.now() - datetime.timedelta(days=1)
        
        if end_time is None:
            end_time = datetime.datetime.now()
        
        filtered_log = []
        for entry in self.compliance_log:
            entry_time = datetime.datetime.fromisoformat(entry["timestamp"])
            if start_time <= entry_time <= end_time:
                filtered_log.append(entry)
        
        if not filtered_log:
            return {"error": "No compliance data available for the specified time period"}
        
        total_checks = len(filtered_log)
        compliant_checks = sum(1 for entry in filtered_log if entry["compliant"])
        violation_checks = total_checks - compliant_checks
        
        issue_counts = {}
        for entry in filtered_log:
            if not entry["compliant"]:
                for issue in entry["issues"]:
                    if issue in issue_counts:
                        issue_counts[issue] += 1
                    else:
                        issue_counts[issue] = 1
        
        symbol_counts = {}
        for entry in filtered_log:
            symbol = entry["order"]["symbol"]
            if symbol in symbol_counts:
                symbol_counts[symbol] += 1
            else:
                symbol_counts[symbol] = 1
        
        return {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "total_checks": total_checks,
                "compliant_checks": compliant_checks,
                "violation_checks": violation_checks,
                "compliance_rate": compliant_checks / total_checks if total_checks > 0 else 0
            },
            "issues": {
                "by_type": issue_counts,
                "most_common": max(issue_counts.items(), key=lambda x: x[1]) if issue_counts else None
            },
            "symbols": {
                "by_count": symbol_counts,
                "most_checked": max(symbol_counts.items(), key=lambda x: x[1]) if symbol_counts else None
            }
        }
