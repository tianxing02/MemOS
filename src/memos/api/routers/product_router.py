import json
import logging
import traceback

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from memos.api.config import APIConfig
from memos.api.context.dependencies import G, get_g_object
from memos.api.product_models import (
    BaseResponse,
    ChatRequest,
    GetMemoryRequest,
    MemoryCreateRequest,
    MemoryResponse,
    SearchRequest,
    SearchResponse,
    SimpleResponse,
    SuggestionRequest,
    SuggestionResponse,
    UserRegisterRequest,
    UserRegisterResponse,
)
from memos.configs.mem_os import MOSConfig
from memos.mem_os.product import MOSProduct
from memos.memos_tools.notification_service import get_error_bot_function, get_online_bot_function


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/product", tags=["Product API"])

# Initialize MOSProduct instance with lazy initialization
MOS_PRODUCT_INSTANCE = None


def get_mos_product_instance():
    """Get or create MOSProduct instance."""
    global MOS_PRODUCT_INSTANCE
    if MOS_PRODUCT_INSTANCE is None:
        default_config = APIConfig.get_product_default_config()
        logger.info(f"*********init_default_mos_config********* {default_config}")
        from memos.configs.mem_os import MOSConfig

        mos_config = MOSConfig(**default_config)

        # Get default cube config from APIConfig (may be None if disabled)
        default_cube_config = APIConfig.get_default_cube_config()
        logger.info(f"*********initdefault_cube_config******** {default_cube_config}")

        # Get DingDing bot functions
        dingding_enabled = APIConfig.is_dingding_bot_enabled()
        online_bot = get_online_bot_function() if dingding_enabled else None
        error_bot = get_error_bot_function() if dingding_enabled else None

        MOS_PRODUCT_INSTANCE = MOSProduct(
            default_config=mos_config,
            default_cube_config=default_cube_config,
            online_bot=online_bot,
            error_bot=error_bot,
        )
        logger.info("MOSProduct instance created successfully with inheritance architecture")
    return MOS_PRODUCT_INSTANCE


get_mos_product_instance()


@router.post("/configure", summary="Configure MOSProduct", response_model=SimpleResponse)
def set_config(config):
    """Set MOSProduct configuration."""
    global MOS_PRODUCT_INSTANCE
    MOS_PRODUCT_INSTANCE = MOSProduct(default_config=config)
    return SimpleResponse(message="Configuration set successfully")


@router.post("/users/register", summary="Register a new user", response_model=UserRegisterResponse)
def register_user(user_req: UserRegisterRequest, g: Annotated[G, Depends(get_g_object)]):
    """Register a new user with configuration and default cube."""
    try:
        # Set request-related information in g object
        g.user_id = user_req.user_id
        g.action = "user_register"
        g.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Starting user registration for user_id: {user_req.user_id}")
        logger.info(f"Request trace_id: {g.trace_id}")
        logger.info(f"Request timestamp: {g.timestamp}")

        # Get configuration for the user
        user_config, default_mem_cube = APIConfig.create_user_config(
            user_name=user_req.user_id, user_id=user_req.user_id
        )
        logger.info(f"user_config: {user_config.model_dump(mode='json')}")
        logger.info(f"default_mem_cube: {default_mem_cube.config.model_dump(mode='json')}")
        mos_product = get_mos_product_instance()

        # Register user with default config and mem cube
        result = mos_product.user_register(
            user_id=user_req.user_id,
            user_name=user_req.user_name,
            interests=user_req.interests,
            config=user_config,
            default_mem_cube=default_mem_cube,
        )

        if result["status"] == "success":
            return UserRegisterResponse(
                message="User registered successfully",
                data={"user_id": result["user_id"], "mem_cube_id": result["default_cube_id"]},
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])

    except Exception as err:
        logger.error(f"Failed to register user: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.get(
    "/suggestions/{user_id}", summary="Get suggestion queries", response_model=SuggestionResponse
)
def get_suggestion_queries(user_id: str):
    """Get suggestion queries for a specific user."""
    try:
        mos_product = get_mos_product_instance()
        suggestions = mos_product.get_suggestion_query(user_id)
        return SuggestionResponse(
            message="Suggestions retrieved successfully", data={"query": suggestions}
        )
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to get suggestions: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.post(
    "/suggestions",
    summary="Get suggestion queries with language",
    response_model=SuggestionResponse,
)
def get_suggestion_queries_post(suggestion_req: SuggestionRequest):
    """Get suggestion queries for a specific user with language preference."""
    try:
        mos_product = get_mos_product_instance()
        suggestions = mos_product.get_suggestion_query(
            user_id=suggestion_req.user_id, language=suggestion_req.language
        )
        return SuggestionResponse(
            message="Suggestions retrieved successfully", data={"query": suggestions}
        )
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to get suggestions: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.post("/get_all", summary="Get all memories for user", response_model=MemoryResponse)
def get_all_memories(memory_req: GetMemoryRequest):
    """Get all memories for a specific user."""
    try:
        mos_product = get_mos_product_instance()
        if memory_req.search_query:
            result = mos_product.get_subgraph(
                user_id=memory_req.user_id,
                query=memory_req.search_query,
                mem_cube_ids=memory_req.mem_cube_ids,
            )
            return MemoryResponse(message="Memories retrieved successfully", data=result)
        else:
            result = mos_product.get_all(
                user_id=memory_req.user_id,
                memory_type=memory_req.memory_type,
                mem_cube_ids=memory_req.mem_cube_ids,
            )
            return MemoryResponse(message="Memories retrieved successfully", data=result)

    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to get memories: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.post("/add", summary="add a new memory", response_model=SimpleResponse)
def create_memory(memory_req: MemoryCreateRequest):
    """Create a new memory for a specific user."""
    try:
        mos_product = get_mos_product_instance()
        mos_product.add(
            user_id=memory_req.user_id,
            memory_content=memory_req.memory_content,
            messages=memory_req.messages,
            doc_path=memory_req.doc_path,
            mem_cube_id=memory_req.mem_cube_id,
            source=memory_req.source,
            user_profile=memory_req.user_profile,
        )
        return SimpleResponse(message="Memory created successfully")

    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to create memory: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.post("/search", summary="Search memories", response_model=SearchResponse)
def search_memories(search_req: SearchRequest):
    """Search memories for a specific user."""
    try:
        mos_product = get_mos_product_instance()
        result = mos_product.search(
            query=search_req.query,
            user_id=search_req.user_id,
            install_cube_ids=[search_req.mem_cube_id] if search_req.mem_cube_id else None,
            top_k=search_req.top_k,
        )
        return SearchResponse(message="Search completed successfully", data=result)

    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to search memories: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.post("/chat", summary="Chat with MemOS")
def chat(chat_req: ChatRequest):
    """Chat with MemOS for a specific user. Returns SSE stream."""
    try:
        mos_product = get_mos_product_instance()

        def generate_chat_response():
            """Generate chat response as SSE stream."""
            try:
                # Directly yield from the generator without async wrapper
                yield from mos_product.chat_with_references(
                    query=chat_req.query,
                    user_id=chat_req.user_id,
                    cube_id=chat_req.mem_cube_id,
                    history=chat_req.history,
                    internet_search=chat_req.internet_search,
                )

            except Exception as e:
                logger.error(f"Error in chat stream: {e}")
                error_data = f"data: {json.dumps({'type': 'error', 'content': str(traceback.format_exc())})}\n\n"
                yield error_data

        return StreamingResponse(
            generate_chat_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            },
        )

    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to start chat: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.get("/users", summary="List all users", response_model=BaseResponse[list])
def list_users():
    """List all registered users."""
    try:
        mos_product = get_mos_product_instance()
        users = mos_product.list_users()
        return BaseResponse(message="Users retrieved successfully", data=users)
    except Exception as err:
        logger.error(f"Failed to list users: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.get("/users/{user_id}", summary="Get user info", response_model=BaseResponse[dict])
async def get_user_info(user_id: str):
    """Get user information including accessible cubes."""
    try:
        mos_product = get_mos_product_instance()
        user_info = mos_product.get_user_info(user_id)
        return BaseResponse(message="User info retrieved successfully", data=user_info)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to get user info: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.get(
    "/configure/{user_id}", summary="Get MOSProduct configuration", response_model=SimpleResponse
)
def get_config(user_id: str):
    """Get MOSProduct configuration."""
    global MOS_PRODUCT_INSTANCE
    config = MOS_PRODUCT_INSTANCE.default_config
    return SimpleResponse(message="Configuration retrieved successfully", data=config)


@router.get(
    "/users/{user_id}/config", summary="Get user configuration", response_model=BaseResponse[dict]
)
def get_user_config(user_id: str):
    """Get user-specific configuration."""
    try:
        mos_product = get_mos_product_instance()
        config = mos_product.get_user_config(user_id)
        if config:
            return BaseResponse(
                message="User configuration retrieved successfully",
                data=config.model_dump(mode="json"),
            )
        else:
            raise HTTPException(
                status_code=404, detail=f"Configuration not found for user {user_id}"
            )
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to get user config: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.put(
    "/users/{user_id}/config", summary="Update user configuration", response_model=SimpleResponse
)
def update_user_config(user_id: str, config_data: dict):
    """Update user-specific configuration."""
    try:
        mos_product = get_mos_product_instance()

        # Create MOSConfig from the provided data
        config = MOSConfig(**config_data)

        # Update the configuration
        success = mos_product.update_user_config(user_id, config)
        if success:
            return SimpleResponse(message="User configuration updated successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to update user configuration")

    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(traceback.format_exc())) from err
    except Exception as err:
        logger.error(f"Failed to update user config: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.get(
    "/instances/status", summary="Get user configuration status", response_model=BaseResponse[dict]
)
def get_instance_status():
    """Get information about active user configurations in memory."""
    try:
        mos_product = get_mos_product_instance()
        status_info = mos_product.get_user_instance_info()
        return BaseResponse(
            message="User configuration status retrieved successfully", data=status_info
        )
    except Exception as err:
        logger.error(f"Failed to get user configuration status: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err


@router.get("/instances/count", summary="Get active user count", response_model=BaseResponse[int])
def get_active_user_count():
    """Get the number of active user configurations in memory."""
    try:
        mos_product = get_mos_product_instance()
        count = mos_product.get_active_user_count()
        return BaseResponse(message="Active user count retrieved successfully", data=count)
    except Exception as err:
        logger.error(f"Failed to get active user count: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(traceback.format_exc())) from err
