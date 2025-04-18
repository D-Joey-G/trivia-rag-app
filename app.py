import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import json
import os
from typing import List, Dict, Any, Union
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Verify secrets exist
_ = st.secrets["anthropic"]
_ = st.secrets["anthropic"]["api_key"]

# Configuration
ANTHROPIC_API_KEY = st.secrets["anthropic"]["api_key"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight but effective model
DATA_PATH = "trivia_data.json"

# Initialise Anthropic
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

class TriviaKnowledgeSystem:
    def __init__(self, data_source: Union[str, List[Dict[str, Any]]]):
        """Initialise the trivia knowledge system with data from a file path or direct data."""
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.trivia_data = (
            self._load_data(data_source) if isinstance(data_source, str)
            else data_source
        )
        self.index, self.id_to_data = self._create_index()
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load and parse the trivia data from JSON file."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Data file not found: {data_path}")
            return []
        except json.JSONDecodeError:
            st.error(f"Invalid JSON format in data file: {data_path}")
            return []
    
    def _create_index(self):
        """Create a nearest neighbors index from the trivia data."""
        if not self.trivia_data:
            return None, {}
        
        # Prepare text for embeddings
        texts = []
        id_to_data = {}
        
        for i, item in enumerate(self.trivia_data):
            # Combine relevant fields for embedding
            text = f"{item.get('Question', '')} {item.get('Answer', '')} {item.get('Category', '')}"
            texts.append(text)
            
            # Store the mapping from index to original data
            id_to_data[i] = item
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Normalise embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # Create NearestNeighbors index
        index = NearestNeighbors(n_neighbors=min(len(embeddings), 25), metric='cosine')
        index.fit(embeddings)
        
        return index, id_to_data
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for the most relevant trivia questions to the query."""
        if not self.index:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Normalise for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1)[:, np.newaxis]
        
        # Search the index
        distances, indices = self.index.kneighbors(query_embedding, n_neighbors=min(k, self.index.n_neighbors))
        
        # Convert distances to similarity scores (cosine distance to similarity)
        similarities = 1 - distances[0]
        
        # Get the corresponding trivia items
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx >= 0 and similarity > 0:  # Valid index and positive similarity
                item = self.id_to_data.get(idx, {}).copy()
                item['similarity_score'] = float(similarity)
                results.append(item)
        
        return results
    
    def generate_synthesis(self, query: str, related_items: List[Dict[str, Any]], use_world_knowledge: bool = False) -> str:
        """Generate a synthesised overview of the related trivia items."""
        if not related_items or not ANTHROPIC_API_KEY:
            return ""
        
        # Format the related questions for the prompt
        formatted_items = []
        context_facts = []
        for item in related_items:
            formatted_item = (
                f"Question: {item.get('Question', 'N/A')}\n"
                f"Answer: {item.get('Answer', 'N/A')}\n"
                f"Category: {item.get('Category', 'N/A')}"
            )
            formatted_items.append(formatted_item)
            context_facts.append(f"- {item.get('Question', 'N/A')} {item.get('Answer', 'N/A')}")
        
        try:
            if use_world_knowledge:
                # Use Sonnet 3.5 with world knowledge
                prompt_instruction = f"""
                You are an expert educator creating an in-depth educational synthesis about "{query}" with aim of helping someone win trivia competitions.
                Below are some relevant facts retrieved from a trivia database.

                Retrieved Facts:
                {context_facts}

                Your task:
                1. Identify the core concepts and entities in these facts
                2. Expand significantly on each concept using your broader knowledge
                3. Explain underlying principles, historical context, and practical applications
                4. Draw explicit connections between these concepts showing how they relate
                5. Include relevant examples and analogies not mentioned in the facts
                6. Address common misconceptions or particularly challenging aspects
                7. Structure your response with clear markdown formatting:
                   - Use ## for main section headers
                   - Use ### for subsection headers
                   - Put key terms, names, and important concepts in **bold**
                   - Use bullet points for lists of related facts
                8. Ensure the final output is well-organised with clear visual hierarchy

                Format your response in markdown with proper headers and emphasis.
                """
                system_message = "You are an expert educator specializing in making complex subjects accessible and interconnected. You excel at explaining the 'why' behind facts, drawing connections between concepts, and providing rich context that helps information stick. Your goal is to transform isolated facts into cohesive knowledge frameworks. Focus on coherence and accuracy."
                
                # Generate the synthesis using Claude Sonnet
                response = anthropic.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=2500,
                    temperature=0.7,
                    system=system_message,
                    messages=[
                        {"role": "user", "content": prompt_instruction}
                    ]
                )
            else:
                # Create the standard prompt without world knowledge
                prompt = f"""
                Based on the following trivia questions and answers about "{query}":
                
                {"\n\n".join(formatted_items)}
                
                Create a coherent, informative synthesis that organises this information 
                into an educational overview. Include all key facts while creating natural 
                transitions between topics. Format your response using markdown:
                - Use ## for main section headers
                - Use ### for subsection headers
                - Put key terms, names, and important concepts in **bold**
                - Use bullet points for lists of related facts
                Ensure the content flows naturally while maintaining clear visual hierarchy.
                """
                
                # Generate the synthesis using Claude Haiku 3, which is cheaper but still effective for the simple task
                response = anthropic.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    temperature=0.7,
                    system="You are an educational content creator specializing in creating concise, informative summaries from trivia facts.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            
            return response.content[0].text.strip()
        except Exception as e:
            st.error(f"Error generating synthesis: {str(e)}")
            return ""

def main():
    st.set_page_config(
        page_title="Trivia Knowledge System",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Trivia Knowledge Retrieval System")
    
    # Initialise
    knowledge_system = TriviaKnowledgeSystem(DATA_PATH)
    
    if not knowledge_system.trivia_data:
        st.error("Failed to load trivia data. Please check the repository includes trivia_data.json")
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.title("Settings")
    num_results = st.sidebar.slider("Number of results", 5, 25, 15)
    enable_synthesis = st.sidebar.checkbox("Enable LLM Synthesis", value=True)
    
    use_world_knowledge = False
    if enable_synthesis:
        use_world_knowledge = st.sidebar.checkbox("Use LLM's world knowledge", value=False)
    
    if not ANTHROPIC_API_KEY and enable_synthesis:
        st.sidebar.warning("Anthropic API key not set. Synthesis will be disabled.")
        enable_synthesis = False
    
    # Search input
    query = st.text_input("Enter your query:", placeholder="e.g., Ancient Greek philosophers")
    
    # Process search when query is submitted
    if query:
        with st.spinner("Searching for relevant questions..."):
            related_items = knowledge_system.search(query, k=num_results)
        
        if related_items:
            st.success(f"Found {len(related_items)} related questions")
            
            # Display the related questions
            st.subheader("Related Quiz Questions")
            for i, item in enumerate(related_items):
                with st.expander(f"{i+1}. {item.get('Question', 'N/A')}"):
                    st.write(f"**Answer:** {item.get('Answer', 'N/A')}")
                    st.write(f"**Category:** {item.get('Category', 'N/A')}")
                    st.write(f"**Similarity Score:** {item.get('similarity_score', 0):.4f}")
            
            # Generate and display synthesis if enabled
            if enable_synthesis:
                with st.spinner("Generating synthesised content..."):
                    synthesis = knowledge_system.generate_synthesis(query, related_items, use_world_knowledge)
                
                if synthesis:
                    st.subheader("Knowledge Synthesis")
                    st.markdown(synthesis)
                else:
                    st.warning("Failed to generate synthesis.")
        else:
            st.warning("No related trivia found for your query.")
    
    # Information about the data
    with st.sidebar.expander("About the Data"):
        st.write(f"Total quiz questions: {len(knowledge_system.trivia_data)}")
        
        # Show categories distribution
        if knowledge_system.trivia_data:
            categories = {}
            for item in knowledge_system.trivia_data:
                cat = item.get('Category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            st.write("**Categories:**")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {cat}: {count}")

if __name__ == "__main__":
    main()
