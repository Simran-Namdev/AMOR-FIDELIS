# facts_knowledge_base.py
import os

class FactsKnowledgeBase:
    def __init__(self, facts_directory):
        self.facts_directory = facts_directory
        self.object_descriptions = self.load_facts()

    def load_facts(self):
        object_descriptions = {}
        for filename in os.listdir(self.facts_directory):
            if filename.endswith(".txt"):
                with open(os.path.join(self.facts_directory, filename), 'r') as file:
                    object_name = os.path.splitext(filename)[0]
                    description = file.read().strip()
                    object_descriptions[object_name] = description
        return object_descriptions

    def get_description(self, object_name):
        description = self.object_descriptions.get(object_name)
        if description:
            return description
        else:
            return "Description not available."