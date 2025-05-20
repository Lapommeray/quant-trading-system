"""
AI Coordinator Module

This module implements the AI Coordinator that orchestrates communication between
different AI components in the QMP Overrider system.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AICoordinator:
    """
    AI Coordinator for routing events and messages between AI components.
    
    This class serves as the central nervous system of the QMP Overrider's AI architecture,
    enabling semantic routing of events and messages between different AI agents.
    """
    
    def __init__(self):
        """Initialize the AI Coordinator with agent workers"""
        self.embedding_model = self._initialize_embedding_model()
        self.agentic_workers = {
            "trading_strategy": {
                "role": "Execute optimal trading strategies based on market conditions",
                "embedding": None
            },
            "risk_management": {
                "role": "Monitor and manage risk exposure across all positions",
                "embedding": None
            },
            "compliance": {
                "role": "Ensure all trading activities comply with regulations",
                "embedding": None
            },
            "market_intelligence": {
                "role": "Analyze market conditions and provide actionable insights",
                "embedding": None
            },
            "quantum_oracle": {
                "role": "Predict future price movements using quantum principles",
                "embedding": None
            }
        }
        
        self._initialize_agent_embeddings()
        
        self.message_history = []
        
        print("AI Coordinator initialized with semantic routing capabilities")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model for semantic routing"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            return TfidfVectorizer(stop_words='english')
    
    def _initialize_agent_embeddings(self):
        """Initialize embeddings for each agent based on their role description"""
        try:
            for agent_id, agent_data in self.agentic_workers.items():
                agent_data["embedding"] = self.embedding_model.encode([agent_data["role"]])[0]
        except AttributeError:
            roles = [agent_data["role"] for agent_data in self.agentic_workers.values()]
            embeddings = self.embedding_model.fit_transform(roles).toarray()
            
            for i, (agent_id, agent_data) in enumerate(self.agentic_workers.items()):
                agent_data["embedding"] = embeddings[i]
    
    def route_event(self, event, source_agent=None):
        """
        Route an event to the most appropriate agent based on semantic similarity
        
        Parameters:
        - event: Dict containing event data including 'message' key
        - source_agent: ID of the agent that generated this event (to avoid routing back)
        
        Returns:
        - Dict with target_agent and routing_confidence
        """
        if 'message' not in event:
            raise ValueError("Event must contain a 'message' key")
        
        try:
            event_embedding = self.embedding_model.encode([event['message']])[0]
        except AttributeError:
            event_embedding = self.embedding_model.transform([event['message']]).toarray()[0]
        
        similarities = {}
        for agent_id, agent_data in self.agentic_workers.items():
            if agent_id != source_agent:  # Don't route back to source
                similarity = cosine_similarity(
                    [event_embedding], 
                    [agent_data["embedding"]]
                )[0][0]
                similarities[agent_id] = similarity
        
        if not similarities:
            return {"target_agent": None, "routing_confidence": 0.0}
            
        target_agent = max(similarities, key=similarities.get)
        routing_confidence = similarities[target_agent]
        
        self.message_history.append({
            "timestamp": event.get('timestamp', None),
            "source": source_agent,
            "message": event['message'],
            "target": target_agent,
            "confidence": routing_confidence
        })
        
        return {
            "target_agent": target_agent,
            "routing_confidence": routing_confidence
        }
    
    def broadcast_event(self, event, source_agent=None, min_confidence=0.5):
        """
        Broadcast an event to all relevant agents based on confidence threshold
        
        Parameters:
        - event: Dict containing event data including 'message' key
        - source_agent: ID of the agent that generated this event (to avoid routing back)
        - min_confidence: Minimum confidence threshold for routing
        
        Returns:
        - Dict with target_agents and their routing confidences
        """
        if 'message' not in event:
            raise ValueError("Event must contain a 'message' key")
        
        try:
            event_embedding = self.embedding_model.encode([event['message']])[0]
        except AttributeError:
            event_embedding = self.embedding_model.transform([event['message']]).toarray()[0]
        
        targets = {}
        for agent_id, agent_data in self.agentic_workers.items():
            if agent_id != source_agent:  # Don't route back to source
                similarity = cosine_similarity(
                    [event_embedding], 
                    [agent_data["embedding"]]
                )[0][0]
                
                if similarity >= min_confidence:
                    targets[agent_id] = similarity
        
        self.message_history.append({
            "timestamp": event.get('timestamp', None),
            "source": source_agent,
            "message": event['message'],
            "broadcast_targets": list(targets.keys()),
            "broadcast_confidences": targets
        })
        
        return targets
    
    def register_agent(self, agent_id, role_description):
        """
        Register a new agent with the coordinator
        
        Parameters:
        - agent_id: Unique identifier for the agent
        - role_description: Text description of the agent's role
        
        Returns:
        - True if registration successful, False otherwise
        """
        if agent_id in self.agentic_workers:
            return False
        
        try:
            role_embedding = self.embedding_model.encode([role_description])[0]
        except AttributeError:
            roles = [agent_data["role"] for agent_data in self.agentic_workers.values()]
            roles.append(role_description)
            embeddings = self.embedding_model.fit_transform(roles).toarray()
            role_embedding = embeddings[-1]
        
        self.agentic_workers[agent_id] = {
            "role": role_description,
            "embedding": role_embedding
        }
        
        return True
    
    def get_routing_statistics(self):
        """
        Get statistics about message routing
        
        Returns:
        - Dict with routing statistics
        """
        if not self.message_history:
            return {"message_count": 0}
        
        stats = {
            "message_count": len(self.message_history),
            "agent_activity": {},
            "average_confidence": 0.0
        }
        
        for msg in self.message_history:
            source = msg.get("source", "unknown")
            if source not in stats["agent_activity"]:
                stats["agent_activity"][source] = {"sent": 0, "received": 0}
            stats["agent_activity"][source]["sent"] += 1
            
            if "target" in msg:
                target = msg["target"]
                if target not in stats["agent_activity"]:
                    stats["agent_activity"][target] = {"sent": 0, "received": 0}
                stats["agent_activity"][target]["received"] += 1
            
            if "broadcast_targets" in msg:
                for target in msg["broadcast_targets"]:
                    if target not in stats["agent_activity"]:
                        stats["agent_activity"][target] = {"sent": 0, "received": 0}
                    stats["agent_activity"][target]["received"] += 1
        
        confidence_sum = 0.0
        confidence_count = 0
        
        for msg in self.message_history:
            if "confidence" in msg:
                confidence_sum += msg["confidence"]
                confidence_count += 1
            elif "broadcast_confidences" in msg:
                for conf in msg["broadcast_confidences"].values():
                    confidence_sum += conf
                    confidence_count += 1
        
        if confidence_count > 0:
            stats["average_confidence"] = confidence_sum / confidence_count
        
        return stats
