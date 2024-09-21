import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, TypedDict, List
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, END
import os

from torchvision.models import resnet50, ResNet50_Weights

class AgentState(TypedDict):
    image_features: torch.Tensor
    visual_attributes: List[str]
    attribute_scores: Dict[str, float]
    current_query: str
    reasoning: List[str]
    final_answer: str


class FeatureExtractionAgent:
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def run(self,x, state: AgentState):
        # In a real scenario, you'd load the image from a file or URL
        # Here, we're using a placeholder
        with torch.no_grad():
            state["image_features"] = self.model(x).squeeze().cpu()
        
        state["reasoning"].append("Extracted image features using ResNet50")
        return state
