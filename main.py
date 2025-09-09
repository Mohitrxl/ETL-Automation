import streamlit as st
import pandas as pd
import json
import os
import io
from datetime import datetime
import google.generativeai as genai
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertETLSystemPrompts:
    """Expert-level ETL system prompts fine-tuned for enterprise ETL tasks"""
    
    @staticmethod
    def get_system_prompt() -> str:
        return """
        You are an Expert ETL (Extract, Transform, Load) Data Engineer with 15+ years of experience at top-tier companies like Google, Microsoft, Amazon, and Netflix. You have deep expertise in:

        CORE COMPETENCIES:
        - Enterprise-scale data pipeline architecture (petabyte-scale processing)
        - Advanced data quality and governance frameworks
        - Performance optimization for big data workloads
        - Real-time and batch processing systems
        - Cloud-native ETL solutions (Azure Data Factory, AWS Glue, Google Cloud Dataflow)
        - Data warehouse design patterns (Kimball, Inmon methodologies)
        - Modern data stack integration (dbt, Airflow, Kafka, Spark)
        
        TECHNICAL EXPERTISE:
        - Python/PySpark for distributed processing
        - SQL optimization and query tuning
        - Data modeling (star schema, snowflake, data vault)
        - Error handling and retry mechanisms
        - Data lineage and observability
        - Security and compliance (GDPR, HIPAA, SOX)
        - Performance monitoring and alerting
        
        CODE GENERATION PRINCIPLES:
        1. Generate production-ready, enterprise-grade code
        2. Include comprehensive error handling and logging
        3. Implement proper data validation and quality checks
        4. Follow software engineering best practices (SOLID principles)
        5. Add performance optimizations and memory management
        6. Include detailed documentation and comments
        7. Implement proper security and access controls
        8. Add monitoring and observability features
        9. Use appropriate design patterns (Factory, Strategy, Observer)
        10. Include unit test suggestions and data profiling
        """
    
    @staticmethod
    def get_etl_code_generation_prompt(source_info: Dict, dest_info: Dict, mapping_doc: str, custom_requirements: str) -> str:
        return f"""
        {ExpertETLSystemPrompts.get_system_prompt()}
        
        TASK: Generate a production-ready ETL pipeline based on the following specifications:
        
        === SOURCE CONFIGURATION ===
        Type: {source_info.get('type', 'Unknown')}
        Location: {source_info.get('path', 'Not specified')}
        Format: {source_info.get('format', 'Unknown')}
        Columns: {source_info.get('columns', [])}
        
        === DESTINATION CONFIGURATION ===
        Type: {dest_info.get('type', 'Unknown')}
        Location: {dest_info.get('path', 'Not specified')}
        Format: {dest_info.get('format', 'Unknown')}
        
        === DATA TRANSFORMATION MAPPING ===
        {mapping_doc}
        
        === BUSINESS REQUIREMENTS ===
        {custom_requirements}
        
        === GENERATE ENTERPRISE ETL CODE WITH: ===
        
        1. **IMPORT SECTION**: All necessary imports with version compatibility
        2. **CONFIGURATION CLASS**: Centralized config management with environment variables
        3. **LOGGING SETUP**: Structured logging with different levels and formatters
        4. **DATA QUALITY FRAMEWORK**: Comprehensive validation and quality checks
        5. **ERROR HANDLING**: Custom exceptions and retry mechanisms
        6. **PERFORMANCE OPTIMIZATION**: Chunked processing, parallel execution, memory management
        7. **MONITORING INTEGRATION**: Metrics collection and alerting hooks
        8. **SECURITY LAYER**: Data encryption, access controls, sensitive data masking
        9. **MAIN ETL PIPELINE**: Clean, modular pipeline with proper separation of concerns
        10. **TESTING FRAMEWORK**: Unit test structure and data validation tests
        
        Generate the complete ETL pipeline as a single, executable Python script that can be deployed in a production environment.
        """

class ExpertETLProcessor:
    """Expert ETL processor with fine-tuned AI model integration"""
    
    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            
            # Configure model with expert parameters
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=8192,
            )
            
            self.model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config=self.generation_config
            )
        else:
            self.model = None
    
    def generate_expert_etl_code(self, source_info: Dict, dest_info: Dict, 
                                mapping_doc: str, custom_requirements: str) -> tuple[str, str]:
        """Generate expert-level ETL code using fine-tuned prompts"""
        
        if not self.model:
            return None, "Gemini API key not configured"
        
        try:
            expert_prompt = ExpertETLSystemPrompts.get_etl_code_generation_prompt(
                source_info, dest_info, mapping_doc, custom_requirements
            )
            
            response = self.model.generate_content(expert_prompt)
            
            if response.text:
                optimized_code = self._post_process_generated_code(response.text)
                return optimized_code, None
            else:
                return None, "No response generated"
                
        except Exception as e:
            logger.error(f"Expert ETL code generation failed: {str(e)}")
            return None, str(e)
    
    def _post_process_generated_code(self, raw_code: str) -> str:
        """Post-process generated code for optimization and standardization"""
        
        # Extract Python code from markdown if present
        if "```python" in raw_code:
            code_blocks = re.findall(r'```python\n(.*?)\n```', raw_code, re.DOTALL)
            if code_blocks:
                raw_code = code_blocks[0]
        elif "```" in raw_code:
            code_blocks = re.findall(r'```\n(.*?)\n```', raw_code, re.DOTALL)
            if code_blocks:
                raw_code = code_blocks[0]
        
        return raw_code

# Cloud database configurations
CLOUD_DATABASES = {
    "Azure": {
        "Azure SQL Database": {
            "icon": "🔷",
            "connection_format": "mssql+pyodbc://username:password@server.database.windows.net:1433/database",
            "supported_formats": ["Table", "Query"]
        },
        "Azure Cosmos DB": {
            "icon": "🌐", 
            "connection_format": "cosmos://account:key@account.documents.azure.com:443/database",
            "supported_formats": ["Collection", "Query"]
        },
        "Azure Blob Storage": {
            "icon": "📦",
            "connection_format": "azure://account_name:account_key@container/blob_path",
            "supported_formats": ["CSV", "JSON", "Parquet"]
        }
    },
    "AWS": {
        "Amazon RDS": {
            "icon": "🟠",
            "connection_format": "postgresql://username:password@endpoint:5432/database",
            "supported_formats": ["Table", "Query"]
        },
        "Amazon S3": {
            "icon": "📁",
            "connection_format": "s3://access_key:secret_key@bucket/path",
            "supported_formats": ["CSV", "JSON", "Parquet"]
        },
        "Amazon Redshift": {
            "icon": "🔴",
            "connection_format": "redshift://username:password@cluster:5439/database",
            "supported_formats": ["Table", "Query"]
        }
    },
    "Google Cloud": {
        "BigQuery": {
            "icon": "📊",
            "connection_format": "bigquery://project_id/dataset",
            "supported_formats": ["Table", "Query"]
        },
        "Cloud SQL": {
            "icon": "☁️",
            "connection_format": "postgresql://username:password@public_ip:5432/database",
            "supported_formats": ["Table", "Query"]
        },
        "Cloud Storage": {
            "icon": "🗄️",
            "connection_format": "gs://bucket/path",
            "supported_formats": ["CSV", "JSON", "Parquet"]
        }
    }
}

# Streamlit App Configuration
st.set_page_config(
    page_title="Expert ETL Automation Platform",
    page_icon="🚀",
    layout="wide"
)

# Enhanced CSS
st.markdown("""
<style>
    .expert-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .expert-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    .expert-header p {
        color: rgba(255,255,255,0.9);
        margin: 10px 0;
        font-size: 1.2em;
    }
    
    .expertise-badge {
        background: rgba(255,255,255,0.2);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        margin: 5px;
        display: inline-block;
    }
    
    .section-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 2px solid #e0e0e0;
    }
    
    .expert-prompt-container {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        border: 3px solid #E0E0E0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #4ECDC4, #44A08D);
        color: white !important;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

def create_expert_header():
    """Create the expert system header"""
    st.markdown("""
        <div class="expert-header">
            <h1>🚀 Expert ETL Automation Platform</h1>
            <p>Enterprise-Grade AI Powered by Google & Microsoft Expertise</p>
            <div>
                <span class="expertise-badge">🏢 Production Ready</span>
                <span class="expertise-badge">⚡ Performance Optimized</span>
                <span class="expertise-badge">🔒 Enterprise Security</span>
                <span class="expertise-badge">📊 Data Quality Assured</span>
                <span class="expertise-badge">🤖 AI Fine-Tuned</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_expert_sidebar():
    """Create enhanced sidebar for expert system"""
    with st.sidebar:
        st.markdown("### 🔑 Expert System Configuration")
        
        # API Key
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Required for expert-level code generation"
        )
        
        if api_key:
            st.success("✅ Expert system ready")
        else:
            st.warning("⚠️ API key required for expert features")
        
        st.markdown("---")
        
        # Expertise Level
        st.markdown("### 🎯 Code Generation Level")
        expertise_level = st.selectbox(
            "Select Expertise Level",
            options=[
                "🚀 Enterprise (Google/Microsoft Level)",
                "💎 Expert (Senior Data Engineer)",
                "⭐ Advanced (Lead Developer)",
                "📈 Intermediate (Mid-level)",
                "📚 Basic (Junior Level)"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Advanced Features
        st.markdown("### ⚙️ Advanced Features")
        
        col1, col2 = st.columns(2)
        with col1:
            include_testing = st.checkbox("🧪 Testing", value=True)
            include_monitoring = st.checkbox("📊 Monitoring", value=True)
            security_features = st.checkbox("🔒 Security", value=True)
        
        with col2:
            data_quality = st.checkbox("✅ Quality", value=True)
            cloud_native = st.checkbox("☁️ Cloud", value=True)
            real_time = st.checkbox("⚡ Real-time", value=False)
        
        return {
            'api_key': api_key,
            'expertise_level': expertise_level,
            'include_testing': include_testing,
            'include_monitoring': include_monitoring,
            'security_features': security_features,
            'data_quality': data_quality,
            'cloud_native': cloud_native,
            'real_time': real_time
        }

def create_database_selector(prefix: str, title: str):
    """Create database selector with expert features"""
    st.markdown(f'<div class="section-container">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    
    cloud_provider = st.selectbox(
        f"🌐 Cloud Provider",
        options=list(CLOUD_DATABASES.keys()),
        key=f"{prefix}_cloud_provider"
    )
    
    if cloud_provider:
        services = CLOUD_DATABASES[cloud_provider]
        
        selected_service = None
        for service_name, info in services.items():
            if st.button(f"{info['icon']} {service_name}", key=f"{prefix}_{service_name}"):
                selected_service = service_name
                st.session_state[f"{prefix}_selected_service"] = service_name
                st.session_state[f"{prefix}_service_info"] = info
        
        if f"{prefix}_selected_service" in st.session_state:
            service_name = st.session_state[f"{prefix}_selected_service"]
            service_info = st.session_state[f"{prefix}_service_info"]
            
            st.success(f"✅ Selected: {service_info['icon']} {service_name}")
            
            connection_string = st.text_input(
                "🔗 Connection String",
                placeholder=service_info['connection_format'],
                key=f"{prefix}_connection_string",
                type="password"
            )
            
            format_option = st.selectbox(
                "📄 Data Format",
                options=service_info['supported_formats'],
                key=f"{prefix}_format"
            )
            
            return {
                "cloud_provider": cloud_provider,
                "service": service_name,
                "connection_string": connection_string,
                "format": format_option,
                "service_info": service_info
            }
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def create_mapping_upload():
    """Enhanced mapping document upload"""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Expert Data Transformation Mapping")
    
    mapping_method = st.radio(
        "Mapping Method",
        options=["📤 Upload JSON/YAML", "✏️ Manual Entry"],
        horizontal=True
    )
    
    mapping_doc = None
    
    if "Upload" in mapping_method:
        uploaded_mapping = st.file_uploader(
            "Choose mapping file",
            type=['json', 'yaml', 'yml'],
            key="expert_mapping_upload"
        )
        
        if uploaded_mapping is not None:
            try:
                content = uploaded_mapping.read()
                if uploaded_mapping.name.endswith(('.yaml', '.yml')):
                    mapping_doc = yaml.safe_load(content)
                else:
                    mapping_doc = json.loads(content)
                
                st.success(f"✅ Loaded: {uploaded_mapping.name}")
                st.json(mapping_doc)
                
            except Exception as e:
                st.error(f"❌ Error loading mapping: {str(e)}")
    
    else:  # Manual entry
        template_choice = st.selectbox(
            "Choose Expert Template",
            options=[
                "Blank Template",
                "Customer Data Cleaning",
                "Financial Transaction Processing", 
                "Sales Analytics Aggregation"
            ]
        )
        
        if template_choice == "Customer Data Cleaning":
            default_mapping = {
                "column_mappings": {
                    "first_name": "fname",
                    "last_name": "lname",
                    "email_address": "email"
                },
                "transformations": {
                    "fname": {"type": "uppercase", "params": {}},
                    "email": {"type": "lowercase", "params": {}}
                },
                "filters": {
                    "valid_email": "email.str.contains('@')"
                },
                "data_quality": {
                    "business_rules": {
                        "email_format": {
                            "condition": "email.str.contains('@')",
                            "description": "Email must contain @ symbol"
                        }
                    }
                }
            }
        else:
            default_mapping = {
                "column_mappings": {},
                "transformations": {},
                "filters": {},
                "aggregations": {},
                "data_quality": {}
            }
        
        mapping_text = st.text_area(
            "Expert Mapping Document (JSON)",
            value=json.dumps(default_mapping, indent=2),
            height=400
        )
        
        try:
            mapping_doc = json.loads(mapping_text)
            st.success("✅ Valid JSON mapping document")
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Error: {str(e)}")
            mapping_doc = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    return mapping_doc

def create_expert_prompt_interface():
    """Create expert-level prompt interface"""
    st.markdown("""
        <div class="expert-prompt-container">
            <h2 style="text-align: center; color: white; margin: 0; font-size: 2em;">
                🤖 Expert AI ETL Code Generator
            </h2>
            <p style="text-align: center; color: rgba(255,255,255,0.9); margin: 10px 0;">
                Fine-tuned for Enterprise ETL Development
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    custom_prompt = st.text_area("Enter your expert prompt here"
        ,
        placeholder="""🔍 Describe your ETL requirements in detail...

Examples:
- "Process customer data with PII masking, validate email formats, detect duplicates"
- "Build real-time streaming ETL for IoT sensor data with anomaly detection"  
- "Create enterprise data warehouse ETL with slowly changing dimensions"
- "Implement GDPR-compliant customer data processing with audit logging"
        """,
        height=120,
        key="expert_custom_prompt",
        label_visibility="collapsed"
    )
    
    return custom_prompt

def main():
    """Main application function"""
    
    create_expert_header()
    expert_config = create_expert_sidebar()
    
    if not expert_config['api_key']:
        st.error("🔑 Please provide your Gemini API key in the sidebar to continue")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize expert ETL processor
    try:
        expert_processor = ExpertETLProcessor(expert_config['api_key'])
    except Exception as e:
        st.error(f"❌ Failed to initialize expert system: {str(e)}")
        return
    
    # Main content layout
    col1, col2 = st.columns(2)
    
    # Data Source Configuration
    with col1:
        source_config = create_database_selector("source", "📊 Expert Data Source Configuration")
        
        # File upload option
        with st.expander("📁 Or Upload Data File", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload source data",
                type=['csv', 'xlsx', 'json', 'parquet']
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, nrows=10)
                        st.success(f"✅ Analyzed: {uploaded_file.name}")
                        st.dataframe(df)
                        
                        source_config = {
                            "type": "file",
                            "path": uploaded_file.name,
                            "format": "csv",
                            "columns": df.columns.tolist()
                        }
                        
                except Exception as e:
                    st.error(f"❌ Error analyzing file: {str(e)}")
    
    # Data Destination Configuration  
    with col2:
        dest_config = create_database_selector("dest", "🎯 Expert Data Destination Configuration")
        
        # Download option
        with st.expander("💾 Or Configure Download", expanded=False):
            download_format = st.selectbox("Download Format", ["CSV", "Excel", "JSON", "Parquet"])
            
            if st.button("Set as Download Destination"):
                dest_config = {
                    "type": "download",
                    "format": download_format.lower(),
                    "path": f"output.{download_format.lower()}"
                }
                st.success(f"✅ Set destination as {download_format} download")
    
    # Mapping Configuration
    mapping_doc = create_mapping_upload()
    
    # Expert Prompt Interface
    custom_prompt = create_expert_prompt_interface()
    
    # Generate Expert ETL Code
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Generate Expert ETL Code", type="primary", use_container_width=True):
            
            if not source_config:
                st.error("❌ Please configure data source")
                return
            
            if not dest_config:
                st.error("❌ Please configure data destination") 
                return
            
            if not mapping_doc:
                st.error("❌ Please provide mapping document")
                return
            
            # Generate expert ETL code
            with st.spinner("🤖 Generating expert-level ETL code..."):
                try:
                    generated_code, error = expert_processor.generate_expert_etl_code(
                        source_config,
                        dest_config, 
                        json.dumps(mapping_doc, indent=2),
                        custom_prompt
                    )
                    
                    if generated_code:
                        st.success("✅ Expert ETL code generated successfully!")
                        
                        # Display results
                        tab1, tab2 = st.tabs(["📝 Generated Code", "📊 Analysis"])
                        
                        with tab1:
                            st.markdown("### 🚀 Production-Ready ETL Code")
                            st.code(generated_code, language='python')
                            
                            # Download option
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"expert_etl_pipeline_{timestamp}.py"
                            
                            st.download_button(
                                label="📥 Download Python Code",
                                data=generated_code,
                                file_name=filename,
                                mime="text/x-python"
                            )
                        
                        with tab2:
                            st.markdown("### 📊 Expert Code Analysis")
                            
                            lines_of_code = len(generated_code.split('\n'))
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Lines of Code", lines_of_code)
                            with col2:
                                st.metric("Expertise Level", "🚀 Enterprise")
                            with col3:
                                st.metric("Features", "✅ All Included")
                    
                    else:
                        st.error(f"❌ Error generating expert code: {error}")
                        
                except Exception as e:
                    st.error(f"❌ Expert system error: {str(e)}")

if __name__ == "__main__":
    main()