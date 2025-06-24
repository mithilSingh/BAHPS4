import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.agents.agent import AgentExecutor
from langchain.schema.agent import AgentAction, AgentFinish
from sentinel_hub_tools import extract_bbox, get_bbox, create_payload, process_request
from sentinel_hub_tools import ProcessRequest, CreatePayload, GetBox, ExtractBox
from app import (
    LoadVector, ReprojectVector, BufferShape, UnionShape, ClipRaster, CalculateSlope, 
    CalculateNDVI, ClassifyFlood, RasterStats, ExtractBbox, SaveGeojson,
    load_vector, reproject_vector, buffer_shape, union_shapes, intersect_shapes,
    distance_between_shapes, clip_raster, calculate_slope,
    calculate_ndvi, classify_flood_risk, raster_stats, extract_bbox_from_geojson,
    save_geojson ,AnalyzeFloodZones , analyze_flood_zones
)

# ------------------------- Tools -------------------------
Tools = [
    StructuredTool.from_function(name="load_vector", func=load_vector, description="Load vector file as geojson and return GeoJSON", args_schema=LoadVector),
    StructuredTool.from_function(name="reproject_vector", func=reproject_vector, description="Reproject a vector file to a given EPSG code", args_schema=ReprojectVector),
    StructuredTool.from_function(name="buffer_shape", func=buffer_shape, description="Buffer geometries in a vector file", args_schema=BufferShape),
    StructuredTool.from_function(name="union_shapes", func=union_shapes, description="Union of two vector files", args_schema=UnionShape),
    StructuredTool.from_function(name="intersect_shapes", func=intersect_shapes, description="Intersection of two vector files", args_schema=UnionShape),
    StructuredTool.from_function(name="distance_between_shapes", func=distance_between_shapes, description="Minimum distance between features in two vector files", args_schema=UnionShape),
    StructuredTool.from_function(name="clip_raster", func=clip_raster, description="Clip a raster with a vector file", args_schema=ClipRaster),
    StructuredTool.from_function(name="calculate_slope", func=calculate_slope, description="Calculate slope from DEM using WhiteboxTools", args_schema=CalculateSlope),
    StructuredTool.from_function(name="calculate_ndvi", func=calculate_ndvi, description="Compute NDVI from NIR and Red band raster file paths", args_schema=CalculateNDVI),
    StructuredTool.from_function(name="classify_flood_risk", func=classify_flood_risk, description="Classify flood risk using NDVI and DEM raster file paths", args_schema=ClassifyFlood),
    StructuredTool.from_function(name="raster_stats", func=raster_stats, description="Compute statistics of a raster", args_schema=RasterStats),
    StructuredTool.from_function(name="extract_bbox_from_geojson", func=extract_bbox_from_geojson, description="Extract bounding box from GeoJSON", args_schema=ExtractBbox),
    StructuredTool.from_function(name="save_geojson", func=save_geojson, description="Save a GeoDataFrame to GeoJSON file", args_schema=SaveGeojson),
    StructuredTool.from_function(name="extract_bbox", func=extract_bbox, description="Extract bounding box for location using OSM", args_schema=ExtractBox),
    StructuredTool.from_function(name="get_bbox", func=get_bbox, description="Get bounding box from vector file", args_schema=GetBox),
    StructuredTool.from_function(name="create_payload", func=create_payload, description="Create Sentinel Hub Process API payload", args_schema=CreatePayload),
    StructuredTool.from_function(name="process_request", func=process_request, description="Provide payload and filepath to request and download raster data from Sentinel Hub (DEM, NDVI, NDWI, etc)", args_schema=ProcessRequest),
    StructuredTool.from_function(name="analyze_flood_zones", func=analyze_flood_zones, description="Analyze flood zones from GeoJSON file", args_schema=AnalyzeFloodZones)
]


# 🧭 System message to control agent behavior
system_prompt = """
You are a geospatial analysis assistant.

Your goal is to autonomously complete geospatial tasks using the available tools. Always use tools in the correct order to achieve the final result.

🔁 Reuse of previous outputs:
- When a tool returns a payload, bounding box, or file path, store it mentally.
- Reuse a previous output **only** if the next tool requires it (e.g., payload → process_request, filepath → clip_raster).
- Do not pass previous outputs unless they are clearly required by the next tool.
- Do not include unrelated previous tool outputs in arguments.

📁 Filenames:
- Do not ask the user for filenames.
- Always generate internal filenames like 'output.tif', 'ndvi.tif', 'flood_map.geojson'.

✅ Dataset types:
- When creating a Sentinel Hub payload, only use dataset types: 'dem', 'ndvi', 'ndwi', 'landcover', 'soil_saturation', 'aod'.
- Do not use unsupported types like 'S1_GRD'.

🧠 Behavior:
- Do not stop midway.
- Do not explain tool calls unless asked.
- Always continue the analysis pipeline until the task is fully complete.

**SPECIAL DIRECTION**
-Donot provide only the filepath
-While calling process_request, you must always provide the payload, filepath.
-Always assume that the last known location remains the context for follow-up queries unless user explicitly changes it.
-Never switch to a different geographic location unless clearly stated in the user query.

"""

from langchain.memory import ConversationBufferWindowMemory

# 🧠 Memory that keeps only last 3 exchanges to stay within token limits
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True
)


# 🔮 LLM setup
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ⚙️ Agent with system prompt
agent = initialize_agent(
    tools=Tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_prompt},
    memory = memory
)

# 🧪 Run + Show internal steps
def debug_run(user_input: str):
    print(f"\n🔍 Prompt: {user_input}\n")
    output = agent.invoke({"input": user_input})

    # LangChain shows tool usage in verbose=True but doesn't separate thoughts
    # So we just wrap this around agent.invoke to track flow

    print("\n✅ Final Answer:\n", output["output"])

# 🚀 CLI
if __name__ == "__main__":
    while True:
        try:
            user_input = input("\n🧑 User: ")
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            debug_run(user_input)
        except Exception as e:
            print("❌ Error:", e)