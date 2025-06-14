from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from textstat import flesch_reading_ease

# Initialize Ollama LLM
llm = Ollama(model="llama3.2")

# Global variable to store current dataframe
current_data = None

# ===============================
# TOOL FUNCTIONS
# ===============================

def analyze_csv(file_path: str) -> str:
    """Analyze CSV file and return insights"""
    global current_data
    file_path = file_path.strip("'\"")
    try:
        # Load data
        df = pd.read_csv(file_path)
        current_data = df
        
        # Get basic info
        rows, cols = df.shape
        missing_vals = df.isnull().sum().sum()
        missing_percent = (missing_vals / (rows * cols)) * 100
        
        # Column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Create summary
        result = f"""
📊 CSV ANALYSIS COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Dataset Info:
   • Rows: {rows:,}
   • Columns: {cols}
   • Missing data: {missing_vals} cells ({missing_percent:.1f}%)

🔢 Numeric Columns ({len(numeric_cols)}):
   {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}

📝 Text Columns ({len(text_cols)}):
   {', '.join(text_cols[:3])}{'...' if len(text_cols) > 3 else ''}

✅ Data Quality: {'Good' if missing_percent < 5 else 'Needs Cleaning'}

💾 Data loaded successfully! Ready for visualization and analysis.
        """
        return result.strip()
        
    except Exception as e:
        return f"❌ Error loading CSV: {str(e)}"


def query_database(question: str, db_path: str = "sample.db") -> str:
    """Query database using natural language"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Simple query mapping (expand this for production)
        if "tables" in question.lower():
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif "customers" in question.lower():
            query = "SELECT * FROM customers LIMIT 5"
        elif "orders" in question.lower():
            query = "SELECT * FROM orders LIMIT 5"
        else:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        result = f"""
🗄️ DATABASE QUERY RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SQL: {query}

📊 Results ({len(df)} rows):
{df.to_string(index=False, max_rows=10)}
        """
        return result
        
    except Exception as e:
        return f"❌ Database error: {str(e)}"


def create_chart(chart_request: str) -> str:
    """Create visualization from loaded data"""
    global current_data
    
    if current_data is None:
        return "❌ No data loaded. Please analyze a CSV file first!"
    
    try:
        df = current_data
        plt.figure(figsize=(10, 6))
        plt.style.use('default')
        
        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols:
            return "❌ No numeric data found for visualization"
        
        # Choose chart type based on request
        if "bar" in chart_request.lower() and text_cols and numeric_cols:
            # Bar chart: categorical vs numeric
            cat_col = text_cols[0]
            num_col = numeric_cols[0]
            
            grouped = df.groupby(cat_col)[num_col].mean().head(10)
            plt.bar(range(len(grouped)), grouped.values)
            plt.xticks(range(len(grouped)), grouped.index, rotation=45)
            plt.title(f'Average {num_col} by {cat_col}')
            plt.ylabel(num_col)
            
        elif "hist" in chart_request.lower():
            # Histogram: distribution of numeric column
            num_col = numeric_cols[0]
            plt.hist(df[num_col].dropna(), bins=20, alpha=0.7)
            plt.title(f'Distribution of {num_col}')
            plt.xlabel(num_col)
            plt.ylabel('Frequency')
            
        else:
            # Default: line plot of first numeric column
            num_col = numeric_cols[0]
            plt.plot(df[num_col].dropna())
            plt.title(f'{num_col} Trend')
            plt.ylabel(num_col)
        
        plt.tight_layout()
        filename = "data_chart.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        result = f"""
📊 CHART CREATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Chart Type: {chart_request}
📁 Saved as: {filename}
📈 Data: {len(df)} rows visualized
✅ Chart generated successfully!
        """
        return result
        
    except Exception as e:
        return f"❌ Chart error: {str(e)}"


def summarize_text(text: str) -> str:
    """Summarize long text using LLM"""
    try:
        word_count = len(text.split())
        
        # Create summarization prompt
        prompt = f"""
Please summarize this text concisely:

{text[:1500]}

Provide:
• Main points (3-5 bullet points)
• Key theme in one sentence
• Overall tone/sentiment
        """
        
        summary = llm(prompt)
        
        result = f"""
📝 TEXT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Original: {word_count} words
🤖 AI Summary:

{summary}

📈 Reading Level: {flesch_reading_ease(text):.0f}/100 (Flesch Score)
        """
        return result
        
    except Exception as e:
        return f"❌ Summary error: {str(e)}"


def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text"""
    try:
        # Create sentiment analysis prompt
        prompt = f"""
Analyze the sentiment of this text:

"{text[:800]}"

Provide:
1. Overall sentiment: Positive/Negative/Neutral (with %)
2. Main emotions: list top 2-3 emotions detected
3. Key insight: one sentence explanation
        """
        
        analysis = llm(prompt)
        
        result = f"""
🎭 SENTIMENT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 Text Length: {len(text)} characters
🤖 AI Analysis:

{analysis}

📊 Analysis Method: LLM-based sentiment detection
        """
        return result
        
    except Exception as e:
        return f"❌ Sentiment analysis error: {str(e)}"


# ===============================
# AGENT SETUP
# ===============================

def create_agent():
    """Create and configure the data analysis agent"""
    
    # Define tools
    tools = [
        Tool(
            name="CSV_Analyzer",
            func=analyze_csv,
            description="Analyze a CSV file. Input: file path (e.g., 'data.csv')"
        ),
        Tool(
            name="Database_Query",
            func=query_database,
            description="Query a database. Input: your question about the data"
        ),
        Tool(
            name="Create_Chart", 
            func=create_chart,
            description="Create visualizations. Input: chart type (bar, histogram, line)"
        ),
        Tool(
            name="Text_Summarizer",
            func=summarize_text,
            description="Summarize long text. Input: the text to summarize"
        ),
        Tool(
            name="Sentiment_Analyzer",
            func=analyze_sentiment,
            description="Analyze text sentiment. Input: text to analyze"
        )
    ]
    
    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    # Create agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    """Main execution pipeline"""
    
    print("🚀 Starting Data Analysis Agent...")
    print("=" * 50)
    
    # Initialize agent
    agent = create_agent()
    
    print("✅ Agent Ready!")
    print("\n🛠️ Available Tools:")
    print("   📊 CSV Analysis")
    print("   🗄️ Database Queries") 
    print("   📈 Data Visualization")
    print("   📝 Text Summarization")
    print("   🎭 Sentiment Analysis")
    
    print("\n💡 Example Commands:")
    print("   • 'Analyze sales_data.csv'")
    print("   • 'Create a bar chart'") 
    print("   • 'Summarize this report: [text]'")
    print("   • 'What's the sentiment of: [text]'")
    
    print("\n" + "=" * 50)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n💬 Your question (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("\n🤖 Processing...")
            response = agent.run(user_input)
            print(f"\n📋 Result:\n{response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different question.")


# ===============================
# RUN PROGRAM
# ===============================

if __name__ == "__main__":
    main()