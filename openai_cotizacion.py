import getpass
import os
from dotenv import load_dotenv
import re
import pdfplumber
import docx
from typing import List
import time
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import io
import fitz
from PIL import Image
import base64
# -----------------------------------------------------------------------------
# 1) Configuración
# -----------------------------------------------------------------------------
CONFIG = {
    "historical_data_path": "./historical_docs",  # Carpera con PDFs/Word históricos
    # Si prefieres separar, usa subcarpetas o ajusta la ruta
}

def get_local_time() -> str:
    """Returns formatted local time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#Convierte PDF en imágenes para mayor análisis en la ventana de contexto del LLM.
#Permite ver texto dentro de imágenes y analizar tablas con mayor facilidad
def pdf_page_to_base64(pdf_path: str, page_number: int):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
# -----------------------------------------------------------------------------
# 2) Procesadores de documentos: PDF y Word
# -----------------------------------------------------------------------------
class PDFProcessor:
    """Lee y separa en chunks un archivo PDF."""
    def __init__(self, text_splitter):
        self.text_splitter = text_splitter
    
    def extract_content(self, file_path: str) -> List[Document]:
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extraer texto
                text = page.extract_text() or ""

                # Extraer tablas
                tables = page.extract_tables()
                table_text = "\n".join([str(table) for table in tables]) if tables else ""

                # Unir texto y tablas
                full_content = f"Página {page_num+1}:\n{text}\n\nTablas:\n{table_text}"

                # Dividir en fragmentos
                chunks = self.text_splitter.split_text(full_content)
                
                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_path,
                                "page": page_num + 1,
                                "content_type": "table" if chunk.strip() == table_text else "text"
                            }
                        )
                    )
        return documents

class WordProcessor:
    """Lee y separa en chunks un archivo Word (.docx)."""
    def __init__(self, text_splitter):
        self.text_splitter = text_splitter
    
    def extract_content(self, file_path: str) -> List[Document]:
        doc = docx.Document(file_path)
        
        # Extraer todo el texto de cada párrafo
        full_text = [para.text for para in doc.paragraphs]
        content = "\n".join(full_text)
        
        # Dividir en fragmentos
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "content_type": "word_text"
                    }
                )
            )
        return documents


class CombinedProcessor:
    """Procesador que decide PDF vs Word según la extensión."""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "##", "."]
        )
        self.pdf_processor = PDFProcessor(self.text_splitter)
        self.word_processor = WordProcessor(self.text_splitter)
    
    def extract_content(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.pdf_processor.extract_content(file_path)
        elif ext == ".docx":
            return self.word_processor.extract_content(file_path)
        else:
            raise ValueError(f"No se soporta la extensión: {ext}")

# -----------------------------------------------------------------------------
# 3) Gestión de documentos históricos (PDF y/o Word)
# -----------------------------------------------------------------------------
class HistoricalDocsManager:
    """
    Carga todos los documentos (.pdf o .docx) de la carpeta "historical_data_path"
    y los indexa en un FAISS vector store.
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = None
        self.processor = CombinedProcessor()
        self.index_path = "./vector_store"  # Directory to store the FAISS index
    
    def load_historical_docs(self, force_reload=False):
        """
        Loads vectors from disk if they exist, otherwise creates new ones.
        Args:
            force_reload (bool): If True, rebuilds the vector store even if it exists
        """
        try:
            # Try to load existing vector store
            if not force_reload and os.path.exists(self.index_path):
                time_forced_reload = time.time()
                print("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization = True
                )
                print(f"Tiempo de carga local: {time.time() - time_forced_reload:.2f} segundos")
                return

            time_normal_load = time.time()
            print("Creating new vector store...")
            all_documents = []
            for root, _, files in os.walk(CONFIG['historical_data_path']):
                for file in files:
                    if file.lower().endswith((".pdf", ".docx")):
                        file_path = os.path.join(root, file)
                        print(f"Processing {file}...")
                        docs = self.processor.extract_content(file_path)
                        all_documents.extend(docs)
            
            if all_documents:
                print("Saving vector store")
                self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
                # Save to disk
                os.makedirs(self.index_path, exist_ok=True)
                self.vector_store.save_local(self.index_path)
                print(f"Tiempo de procesamiento: {time.time() - time_normal_load:.2f} segundos")
                print(f"Vector store saved to {self.index_path}")
                
        except Exception as e:
            print(f"Error loading/creating vector store: {str(e)}")
            raise

    
    def get_similar_projects(self, query: str, k: int = 30) -> tuple[List[Document], str]:
        """Retorna los fragmentos más similares a 'query' de los documentos históricos."""
        if self.vector_store:
            similar_docs = self.vector_store.similarity_search(query, k=k)
            for doc in similar_docs:
                if doc.metadata["source"].endswith(".pdf"):
                    similar_images = pdf_page_to_base64(doc.metadata["source"],doc.metadata["page"])
            return (similar_docs,similar_images)
        return []

# -----------------------------------------------------------------------------
# 4) Generador de ofertas con un "prompt" de calidad
# -----------------------------------------------------------------------------
class ProposalGenerator:
    def __init__(self):
        # LLM local (ChatOllama). Ajusta según tu wrapper o modelo.
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

        # (Opcional) Modelo de re-rank
        self.re_rank_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        
        # Aquí definimos nuestras plantillas (prompts) directamente en el código.
        # Si prefieres, podrías cargarlas desde .md en disco.
        self.templates = self._load_inline_templates()
    
    def _load_inline_templates(self):
        """
        Creamos dos plantillas (tecnica, comercial).
        Cada una tiene un 'prompt' con instrucciones detalladas para generar
        una propuesta de alta calidad.
        
        Usamos {{client_context}} y {{historical_context}} que LangChain sustituye 
        al correr el chain.
        """
        templates = {}

        templates["tecnica"] = (
           
            "Debes redactar una propuesta TÉCNICA profesional basándote en:\n\n"
            "1) Información del cliente en formato de texto y de imagen:\n"
            "{client_context}\n\n"
            "{similar_imgs}\n\n"
            "2) Experiencias previas en proyectos similares:\n"
            "{historical_context}\n\n"
            "La información del cliente es una propuesta técnica, pero es simplemente una otorgada por el cliente para especificar los"+
            "productos y servicios que se deben entregar. La propuesta técnica que debes generar ahora es una propuesta técnica completa, "+
            "que incluye todos los detalles necesarios para la ejecución del proyecto y todos los servicios y productos que puedes proporcionar.\n\n"
            "INSTRUCCIONES:\n"
            "-La propuesta debe ser generada en un archivo HTML, de manera tal "+
            "que mantenga una estética similar a la del archivo dado por el cliente"+
            "como 'Información del cliente'. Provee ÚNICAMENTE la propuesta en HTML, dado que debe ser."+
            "mostrada en la interfaz de usuario.\n"
            "- Estructura tu respuesta con claridad y detalle.\n"
            "- Usa un tono profesional y conciso.\n"
            "- Si no encuentras cierta información, haz asunciones razonables.\n"
            "Por favor, genera la propuesta técnica a continuación:\n"
            "- Estos son algunos de los temas típicos que se desarrollan en este tipo de propuesta.\n"
            "Se deben tomar como una guía para los temas que se deben desarrollar en la propuesta requerida ahora.\n"
            "Determina qué requerimientos especificados aquí no deben, necesariamente, incluirse en el documento a generar,\n"
            "basándote en tu experiencia y en los requisitos mencionados en el archivo provisto por el cliente.\n"
            "Los requerimientos que sí se deban desarrollar, descríbelos en detalle, tomando en cuenta tu experiencia\n"
            "en el desarrollo de propuestas similares.\n\n"

                "1. Objetivo de la Propuesta\n"
                "2. Alcance General de Productos y Servicios\n"
                "3. Alcance Especificaciones de productos y servicios\n"
                "   3.1 Dirección y gestión del proyecto\n"
                "       - Planificación\n"
                "       - Ejecución\n"
                "       - Control\n"
                "       - Cierre\n"
                "   3.2 Ingeniería de Detalle\n"
                "       3.2.1 Listado de documentos a entregar (GENERAL)\n"
                "       3.2.2 Listado de documentos a entregar (ESTRUCTURAL/CIVIL)\n"
                "       3.2.3 Listado de documentos a entregar (ELECTRICIDAD)\n"
                "       3.2.4 Listado de documentos a entregar (EQUIPAMIENTO)\n"
                "       3.2.5 Listado de documentos a entregar (CALIDAD)\n"
                "       3.2.6 Listado de documentos a recibir (a enviar por el CLIENTE)\n"
                "       3.2.7 Consideraciones y exclusiones generales para la elaboración de Ingeniería de detalle\n"
                "   3.3 Éste subtítulo depende de la propuesta.\n"
                "       Suele ser, por ejemplo: 'Shelter Eléctrico Modular', 'Shelter Eléctrico de MT', 'Sala eléctrica' de alta potencia,\n"
                "       entre otras opciones que dependen de para qué sea la propuesta.\n"
                "       3.3.1 Dimensiones Generales\n"
                "       3.3.2 Base de piso\n"
                "       3.3.3 Estructura/Paredes\n"
                "       3.3.4 Estructura/Techo y Sobretecho\n"
                "       3.3.5 Sistema de Izaje\n"
                "       3.3.6 Revestimiento\n"
                "       3.3.7 Aislación térmica\n"
                "       3.3.8 Piso\n"
                "       3.3.9 Aberturas\n"
                "       3.3.10 Plataformas, barandas y escaleras\n"
                "       3.3.11 Esquema de Pintura\n"
                "       3.3.12 Instalación Eléctrica\n"
                "       3.3.13 Climatización\n"
                "       3.3.14 Sistema de Detección y extinción de incendios\n"
                "       3.3.15 Celdas de Media Tensión\n"
                "       3.3.16 Botiquín y Portaplanos\n"
                "       3.3.17 Pruebas\n"
                "       3.3.18 Sistema CCTV\n"
                "   3.4 Tablero de distribución de potencia\n"
                "   3.5 Sistemas de alimentación ininterrumpida\n\n"

                "4. Prestación de servicios\n"
                "   4.1 Montaje e interconexión de equipos y gabinetes eléctricos en Shelter\n"
                "   4.2 Suministro de materiales\n"
                "   4.3 Inspecciones y pruebas\n"
                "   4.4 Servicios en sitio\n\n"

                "5. Transporte\n"
                "6. Garantía\n"
                "7. Productos y servicios no incluidos en la propuesta\n"
                        )

        templates["comercial"] = (
            "Eres un experto en ventas y generación de propuestas COMERCIALES. "
            "Debes redactar una oferta comercial basándote en:\n\n"
            "1) Información del cliente:\n"
            "{client_context}\n\n"
            "2) Experiencias previas en propuestas y proyectos similares:\n"
            "{historical_context}\n\n"
            "INSTRUCCIONES:\n"
            "- Estructura tu respuesta con claridad y detalle.\n"
            "- Incluye detalles de costos, formas de pago, beneficios y diferenciadores.\n"
            "- Usa un tono cordial y persuasivo.\n"
            "- Si faltan detalles, puedes asumirlos razonablemente.\n"
            "Por favor, genera la propuesta comercial a continuación:\n"
        )

        return templates
    
    def generate_proposal(
        self,
        client_docs: List[Document],
        historical_context: str,
        proposal_type: str,
        similar_imgs: str
    ) -> str:
        """
        Genera una propuesta (técnica o comercial) usando la plantilla correspondiente.
        """
        # Unir contenido de documentos del cliente
        client_context = "\n".join([doc.page_content for doc in client_docs])
        
        # Escoger la plantilla según 'proposal_type'
        template_text = self.templates.get(
            proposal_type, 
            "No existe plantilla para este tipo de propuesta."
        )
        
        # Crear prompt dinámico a partir de la plantilla
        prompt = ChatPromptTemplate.from_messages([
        ("system",  "Eres un ingeniero experto en Ingeniería y Gestión de Proyectos. Tus "+
         "tareas más importantes son la confección de propuestas técnicas y comerciales "+
         "para clientes de la empresa. Hoy, debes redactar una propuesta técnica "+
         "para un nuevo cliente. Le das mucha importancia a que el formato de la propuesta"+
         "que generes se parezca al de propuestas que has generado con anterioridad, "+
         "para basarte en tu experiencia y facilitar tu trabajo. Los temas tratados en estas propuestas, "+
         "según tú, se suelen repetir, pero dependen mucho de lo que pida el cliente, "+
         "así que le das especial atención a eso para poder cumplir con sus expectativas."+
         "Eres muy detallista en la elaboración de estas propuestas."),
        ("human", template_text.format(
            similar_imgs = similar_imgs,
            client_context=client_context,
            historical_context=historical_context
        ))
    ])
        
        
        # Crear la cadena de generación con LangChain
        chain = prompt | self.llm | StrOutputParser()
        
        # Ejecutar la cadena
        response = chain.invoke({})
        
        return response

# -----------------------------------------------------------------------------
# 5) Funciones de utilidad
# -----------------------------------------------------------------------------
def clean_proposal(text: str) -> str:
    """Limpieza básica del texto final de la propuesta."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\s*•\s*)', '\n- ', text)
    return text.strip()

def generate_cotizacion(file_paths: List[str], proposal_type: str) -> str:
    """
    Dada una lista de archivos (PDF y/o Word) del cliente y un tipo de propuesta
    ('tecnica' o 'comercial'), genera una cotización usando documentación histórica 
    (también en PDF/Word) como contexto.
    """
    print(f"Starting application at: {get_local_time()}") 
    start_total = time.time()
    start_processing = time.time()
    # 1) Procesar documentos del cliente
    processor = CombinedProcessor()
    client_docs = []
    for fpath in file_paths:
        docs = processor.extract_content(fpath)
        client_docs.extend(docs)
    print(f"1. Tiempo procesamiento documentos: {time.time() - start_processing:.2f} segundos")
    # 2) Cargar documentos históricos y buscar los más relevantes
    start_historical = time.time()
    historical_mgr = HistoricalDocsManager()
    historical_mgr.load_historical_docs()
    
    # Tomamos el texto del cliente para hacer la query al vector store
    client_text = " ".join([doc.page_content for doc in client_docs])
    similar_docs, similar_imgs = historical_mgr.get_similar_projects(client_text, k=5)
    historical_context = "\n".join([doc.page_content for doc in similar_docs])
    print(f"2. Tiempo carga históricos y búsqueda: {time.time() - start_historical:.2f} segundos")
    # 3) Generar propuesta con LLM local
    start_generation = time.time()
    generator = ProposalGenerator()
    proposal = generator.generate_proposal(
        client_docs=client_docs,
        historical_context=historical_context,
        proposal_type=proposal_type,
        similar_imgs = similar_imgs
    )
    print(f"3. Tiempo generación propuesta: {time.time() - start_generation:.2f} segundos")
    # 4) Limpiar y retornar
    start_cleaning = time.time()
    cleaned = clean_proposal(proposal)
    print(f"4. Tiempo limpieza: {time.time() - start_cleaning:.2f} segundos")
    
    print(f"\nTiempo total: {time.time() - start_total:.2f} segundos")
    print(f"Quotation generated at: {get_local_time()}") 
    return cleaned
    


if __name__ == "__main__":
    print(f"Starting application at: {get_local_time()}") 
    
    """
    Suponiendo que tengas una carpeta 'historical_docs/' 
    con PDFs y/o DOCX de proyectos históricos, 
    y en 'file_paths' tengas la ruta(s) del PDF/DOCX del cliente.
    """
   # Load environment variables from .env file
    load_dotenv()

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    # Ejemplo: cambia estas rutas a las de tus archivos reales
    file_paths = ["./propuesta_A.pdf"]
    
    # Selecciona el tipo de propuesta: 'tecnica' o 'comercial'
    proposal_type = "tecnica"

    # Generar la cotización
    cotizacion = generate_cotizacion(file_paths, proposal_type)
    print(f"\nQuotation generated at: {get_local_time()}")
    print("=== PROPUESTA GENERADA ===")
    print(cotizacion)
    print(f"Token usage: {len(cotizacion) // 4}")
    # historical_mgr = HistoricalDocsManager()
    # historical_mgr.load_historical_docs(force_reload=True)