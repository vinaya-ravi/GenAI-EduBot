import redis
import json
import logging
from typing import Any, Dict, List, Optional
from config.redis_config import get_redis_config, REDIS_KEY_PREFIXES

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self):
        """Initialize Redis connection"""
        try:
            config = get_redis_config()
            self.redis_client = redis.Redis(**config)
            self.redis_client.ping()  # Test connection
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def store_conversation(self, conversation_id: str, data: Dict[str, Any]) -> bool:
        """Store conversation data in Redis"""
        try:
            key = f"{REDIS_KEY_PREFIXES['conversation']}{conversation_id}"
            self.redis_client.set(key, json.dumps(data))
            logger.info(f"Stored conversation {conversation_id} in Redis")
            return True
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation data from Redis"""
        try:
            key = f"{REDIS_KEY_PREFIXES['conversation']}{conversation_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving conversation: {str(e)}")
            return None

    def store_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Store user data in Redis"""
        try:
            key = f"{REDIS_KEY_PREFIXES['user']}{user_id}"
            self.redis_client.set(key, json.dumps(data))
            logger.info(f"Stored user data for {user_id} in Redis")
            return True
        except Exception as e:
            logger.error(f"Error storing user data: {str(e)}")
            return False

    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user data from Redis"""
        try:
            key = f"{REDIS_KEY_PREFIXES['user']}{user_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving user data: {str(e)}")
            return None

    def store_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Store agent state in Redis"""
        try:
            key = f"{REDIS_KEY_PREFIXES['agent']}{agent_id}"
            self.redis_client.set(key, json.dumps(state))
            logger.info(f"Stored agent state for {agent_id} in Redis")
            return True
        except Exception as e:
            logger.error(f"Error storing agent state: {str(e)}")
            return False

    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent state from Redis"""
        try:
            key = f"{REDIS_KEY_PREFIXES['agent']}{agent_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving agent state: {str(e)}")
            return None

    def delete_key(self, key: str) -> bool:
        """Delete a key from Redis"""
        try:
            self.redis_client.delete(key)
            logger.info(f"Deleted key {key} from Redis")
            return True
        except Exception as e:
            logger.error(f"Error deleting key: {str(e)}")
            return False

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all stored conversations"""
        try:
            pattern = f"{REDIS_KEY_PREFIXES['conversation']}*"
            keys = self.redis_client.keys(pattern)
            conversations = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    conversations.append(json.loads(data))
            return conversations
        except Exception as e:
            logger.error(f"Error retrieving all conversations: {str(e)}")
            return [] 