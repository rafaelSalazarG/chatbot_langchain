# chatbot_langchain
El código en los archivos app.py y openai_cotizacion.py es parte de una aplicación diseñada para generar propuestas técnicas o comerciales basadas en documentos proporcionados por un cliente y documentación histórica almacenada en el sistema. 

1. Propósito General
La aplicación utiliza procesamiento de documentos (PDF/Word), un modelo de lenguaje (LLM), y un sistema de búsqueda semántica para generar propuestas personalizadas. Estas propuestas se basan en:

Documentos proporcionados por el cliente.
Documentos históricos previamente indexados.
Plantillas predefinidas para propuestas técnicas y comerciales.
El flujo principal incluye:

Procesamiento de documentos del cliente.
Búsqueda de documentos históricos relevantes.
Generación de una propuesta personalizada utilizando un modelo de lenguaje.
2. openai_cotizacion.py
Este archivo contiene la lógica principal de la aplicación, incluyendo la configuración, procesamiento de documentos, gestión de documentos históricos, y generación de propuestas.

Componentes Clave
Procesamiento de Documentos

PDFProcessor y WordProcessor: Clases que extraen contenido de archivos PDF y Word, dividiéndolos en fragmentos manejables (chunks) utilizando un separador configurable.
CombinedProcessor: Decide qué procesador usar según la extensión del archivo y combina los resultados.
Gestión de Documentos Históricos

HistoricalDocsManager:
Carga documentos históricos desde una carpeta (historical_docs).
Indexa los documentos en un vector store (FAISS) utilizando embeddings generados por el modelo sentence-transformers/all-mpnet-base-v2.
Permite realizar búsquedas semánticas para encontrar documentos similares a los proporcionados por el cliente.
Generación de Propuestas

ProposalGenerator:
Utiliza un modelo de lenguaje (Google Generative AI) para generar propuestas basadas en plantillas predefinidas.
Las plantillas están diseñadas para propuestas técnicas y comerciales, con instrucciones detalladas para estructurar el contenido.
Integra información del cliente y documentos históricos relevantes para personalizar la propuesta.
Flujo Principal

La función generate_cotizacion coordina todo el proceso:
Procesa los documentos del cliente.
Busca documentos históricos relevantes.
Genera la propuesta utilizando el modelo de lenguaje.
Limpia y retorna el texto final.
