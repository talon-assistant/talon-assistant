import requests
import json


class LLMClient:
    """KoboldCpp API wrapper for LLM inference"""

    def __init__(self, config):
        llm_config = config["llm"]
        self.endpoint = llm_config["endpoint"]
        self.max_length = llm_config["max_length"]
        self.temperature = llm_config["temperature"]
        self.top_p = llm_config["top_p"]
        self.rep_pen = llm_config["rep_pen"]
        self.timeout = llm_config["timeout"]
        self.stop_sequences = llm_config["stop_sequences"]
        self.prompt_template = llm_config["prompt_template"]

    def test_connection(self):
        """Test connection to KoboldCpp server"""
        try:
            test_response = requests.post(
                self.endpoint,
                json={
                    "prompt": f"{self.prompt_template['user_prefix']}Hello{self.prompt_template['user_suffix']}{self.prompt_template['assistant_prefix']}",
                    "max_length": 10,
                    "stop_sequence": self.stop_sequences
                },
                timeout=10
            )
            if test_response.status_code == 200:
                print("   KoboldCpp connected!")
                return True
            else:
                print(f"   Warning: Status {test_response.status_code}")
                return False
        except Exception as e:
            print(f"   Warning: {e}")
            return False

    def generate(self, prompt, use_vision=False, screenshot_b64=None,
                 max_length=None, system_prompt=None, temperature=None):
        """Send prompt to KoboldCpp and return generated text.

        Args:
            prompt: The user message / content to send.
            use_vision: Whether this is a vision request.
            screenshot_b64: Base64 screenshot for vision.
            max_length: Override max token generation length.
            system_prompt: Optional system message (ChatML <|im_start|>system).
            temperature: Override temperature for this call (lower = more factual).
        """
        effective_max_length = max_length or self.max_length
        effective_temperature = temperature if temperature is not None else self.temperature

        # Build system block if provided
        system_block = ""
        if system_prompt:
            system_block = (
                f"<|im_start|>system\n"
                f"{system_prompt}<|im_end|>\n"
            )

        if use_vision and screenshot_b64:
            vision_prefix = self.prompt_template["vision_prefix"]
            formatted_prompt = (
                f"{system_block}"
                f"{self.prompt_template['user_prefix']}"
                f"{vision_prefix}{prompt}"
                f"{self.prompt_template['user_suffix']}"
                f"{self.prompt_template['assistant_prefix']}"
            )
            payload = {
                "prompt": formatted_prompt,
                "max_length": effective_max_length,
                "temperature": effective_temperature,
                "top_p": self.top_p,
                "rep_pen": self.rep_pen,
                "stop_sequence": self.stop_sequences,
                "images": [screenshot_b64]
            }
        else:
            formatted_prompt = (
                f"{system_block}"
                f"{self.prompt_template['user_prefix']}"
                f"{prompt}"
                f"{self.prompt_template['user_suffix']}"
                f"{self.prompt_template['assistant_prefix']}"
            )
            payload = {
                "prompt": formatted_prompt,
                "max_length": effective_max_length,
                "temperature": effective_temperature,
                "top_p": self.top_p,
                "rep_pen": self.rep_pen,
                "stop_sequence": self.stop_sequences
            }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if 'results' in result and len(result['results']) > 0:
                    return result['results'][0]['text'].strip()
                else:
                    return "Error: Unexpected response format"
            else:
                return f"Error: Status {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"
