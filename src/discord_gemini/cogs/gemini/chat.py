"""Chat flow orchestration for the Gemini cog."""

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from discord import Attachment, Colour, Embed
from discord.commands import ApplicationContext
from google.genai import types

from ...config.auth import ENABLE_CUSTOM_TOOLS, SHOW_COST_EMBEDS
from ...util import (
    MAX_AGENTIC_ITERATIONS,
    TYPING_INDICATOR_INTERVAL,
    AgenticResult,
    ChatCompletionParameters,
    calculate_cost,
    check_mutually_exclusive_tools,
    filter_file_search_incompatible_tools,
    filter_supported_tools_for_model,
    has_server_side_tools,
    model_supports_tool_combinations,
    resolve_tool_name,
    truncate_text,
    validate_builtin_custom_tool_combination,
)
from . import attachments, cache, embeds, responses, state, tooling, usage
from .models import Conversation
from .tool_registry import build_runtime_tool_config, iter_tool_registry
from .views import ButtonView

if TYPE_CHECKING:
    from .cog import GeminiCog
    from .models import Conversation


async def _run_agentic_loop(
    cog: "GeminiCog",
    model: str,
    contents: list[Any],
    config: types.GenerateContentConfig | None,
) -> AgenticResult:
    """Run generate_content in a loop, executing custom function calls."""

    result = AgenticResult(response=None, contents=contents)

    for iteration in range(MAX_AGENTIC_ITERATIONS):
        response = await cog.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        result.response = response
        result.iterations = iteration + 1

        usage_counts = usage.extract_usage_counts(response)
        result.total_input_tokens += usage_counts.input_tokens
        result.total_output_tokens += usage_counts.output_tokens
        result.total_thinking_tokens += usage_counts.thinking_tokens

        function_calls = response.function_calls
        if not function_calls:
            break

        model_parts = responses._get_response_content_parts(response)
        if model_parts:
            contents.append({"role": "model", "parts": model_parts})

        func_response_parts = []
        for function_call in function_calls:
            name = function_call.name or ""
            if not name:
                continue
            cog.logger.info(
                "Agentic loop iter %d: calling %s(%s)",
                iteration + 1,
                name,
                function_call.args,
            )
            result.tool_calls_made.append(name)
            call_result = await tooling.execute_tool_call(name, function_call.args)
            function_response = types.FunctionResponse(
                name=name,
                response=call_result,
            )
            function_call_id = getattr(function_call, "id", None)
            if function_call_id:
                function_response.id = function_call_id
            func_response_parts.append(types.Part(function_response=function_response))

        contents.append({"role": "user", "parts": func_response_parts})
    else:
        cog.logger.warning("Agentic loop hit max iterations (%d)", MAX_AGENTIC_ITERATIONS)

    return result


async def keep_typing(cog: "GeminiCog", channel: Any) -> None:
    """Keep Discord's typing indicator alive for long-running requests."""

    try:
        cog.logger.debug("Starting typing indicator loop in channel %s", channel.id)
        while True:
            async with channel.typing():
                cog.logger.debug("Sent typing indicator to channel %s", channel.id)
                await asyncio.sleep(TYPING_INDICATOR_INTERVAL)
    except asyncio.CancelledError:
        cog.logger.debug("Typing indicator cancelled for channel %s", channel.id)
        raise


def _add_custom_function_tools(
    config_args: dict[str, Any],
    model: str = "",
    custom_functions_enabled: bool = False,
    providers: list[tooling.ToolProvider] | None = None,
) -> None:
    """Attach Python callable tools and keep execution in the manual tool loop."""

    if not (custom_functions_enabled and ENABLE_CUSTOM_TOOLS):
        return

    if not model and providers is None:
        callables = tooling.get_tool_callables()
        if not callables:
            return
        tool_list = list(config_args.get("tools", []))
        tool_list.extend(callables)
        config_args["tools"] = tool_list
        config_args["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )
        return

    active_providers = providers or tooling.get_tool_providers()
    initial_tools = list(config_args.get("tools", []))
    tool_list = list(initial_tools)
    for provider in active_providers:
        if provider.provider_id != "local" or not provider.supports(model):
            continue
        declarations = provider.list_declarations(model)
        if declarations:
            tool_list.extend(declarations)
    if tool_list != initial_tools:
        config_args["tools"] = tool_list
    if any(
        callable(t) or (isinstance(t, dict) and "function_declarations" in t) for t in tool_list
    ):
        config_args["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )


def _configure_tool_context_circulation(
    config_args: dict[str, Any],
    model: str,
    custom_functions_enabled: bool,
) -> None:
    """Enable server-side tool circulation for supported Gemini 3 tool setups."""

    tools = list(config_args.get("tools", []))
    if not tools or not has_server_side_tools(tools) or not model_supports_tool_combinations(model):
        return

    function_calling_config = None
    if custom_functions_enabled:
        function_calling_config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.VALIDATED
        )

    config_args["tool_config"] = types.ToolConfig(
        include_server_side_tool_invocations=True,
        function_calling_config=function_calling_config,
    )


async def handle_new_message_in_conversation(
    cog: "GeminiCog",
    message: Any,
    conversation_wrapper: "Conversation",
) -> None:
    """Handle a new user message inside an existing conversation."""

    params = conversation_wrapper.params
    history = conversation_wrapper.history

    cog.logger.info("Handling new message in conversation %s.", params.conversation_id)
    typing_task = None
    response_embeds: list[Embed] = []

    try:
        if message.author != params.conversation_starter or params.paused:
            return

        typing_task = asyncio.create_task(keep_typing(cog, message.channel))

        user_parts: list[str | dict[str, Any]] = []
        if message.attachments:
            for attachment in message.attachments:
                validation_error = attachments._validate_attachment_size(attachment)
                if validation_error:
                    await message.reply(embed=embeds.build_error_embed(validation_error))
                    return

                attachment_part = await attachments._prepare_attachment_part(
                    cog,
                    attachment,
                    params.uploaded_file_names,
                )
                if attachment_part is not None:
                    user_parts.append(attachment_part)

        if message.content:
            user_parts.append({"text": message.content})
        history.append({"role": "user", "parts": user_parts})

        config_args: dict[str, Any] = {}
        if params.cache_name:
            config_args["cached_content"] = params.cache_name
        elif params.system_instruction:
            config_args["system_instruction"] = params.system_instruction
        if params.temperature is not None:
            config_args["temperature"] = params.temperature
        if params.top_p is not None:
            config_args["top_p"] = params.top_p
        if params.media_resolution is not None:
            config_args["media_resolution"] = params.media_resolution
        thinking_config = responses._build_thinking_config(
            params.thinking_level,
            params.thinking_budget,
        )
        if thinking_config is not None:
            config_args["thinking_config"] = thinking_config
        if params.tools:
            supported_tools, unsupported_tools = filter_supported_tools_for_model(
                params.model,
                params.tools,
            )
            if unsupported_tools:
                cog.logger.warning(
                    "Skipping unsupported tools for model %s: %s",
                    params.model,
                    ", ".join(sorted(set(unsupported_tools))),
                )
            params.tools = supported_tools
            if supported_tools:
                config_args["tools"] = supported_tools

        combination_error = validate_builtin_custom_tool_combination(
            params.model,
            config_args.get("tools", []),
            params.custom_functions_enabled,
        )
        if combination_error:
            await message.reply(embed=embeds.build_error_embed(combination_error))
            return

        _add_custom_function_tools(config_args, params.model, params.custom_functions_enabled)
        _configure_tool_context_circulation(
            config_args, params.model, params.custom_functions_enabled
        )

        history_start = params.cached_history_length if params.cache_name else 0
        contents = [
            {"role": entry["role"], "parts": entry["parts"]} for entry in history[history_start:]
        ]

        generation_config = types.GenerateContentConfig(**config_args) if config_args else None
        pre_loop_len = len(contents)
        try:
            result = await _run_agentic_loop(cog, params.model, contents, generation_config)
        except Exception as cache_error:
            if not params.cache_name:
                raise
            cog.logger.warning("Cached request failed, retrying without cache: %s", cache_error)
            params.cache_name = None
            params.cached_history_length = 0
            config_args.pop("cached_content", None)
            if params.system_instruction:
                config_args["system_instruction"] = params.system_instruction
            contents = [{"role": entry["role"], "parts": entry["parts"]} for entry in history]
            pre_loop_len = len(contents)
            generation_config = types.GenerateContentConfig(**config_args) if config_args else None
            result = await _run_agentic_loop(cog, params.model, contents, generation_config)

        response = result.response
        response_text = response.text or ""
        tool_info = responses.extract_tool_info(response)
        for tool_name in result.tool_calls_made:
            if tool_name not in tool_info["tools_used"]:
                tool_info["tools_used"].append(tool_name)

        if typing_task:
            typing_task.cancel()
            typing_task = None

        if response_text is None:
            response_text = "No response generated by the model."
            cog.logger.warning("Model returned None as response text")

        for entry in contents[pre_loop_len:]:
            history.append(entry)

        response_parts = responses._get_response_content_parts(response)
        history.append({"role": "model", "parts": response_parts or [{"text": response_text}]})

        await cache._maybe_create_cache(cog, params, history, response)

        thinking_text = responses.extract_thinking_text(response)
        embeds.append_thinking_embeds(response_embeds, thinking_text)
        embeds.append_response_embeds(response_embeds, response_text)
        embeds.append_sources_embed(response_embeds, tool_info)

        input_tokens = result.total_input_tokens
        output_tokens = result.total_output_tokens
        thinking_tokens = result.total_thinking_tokens
        maps_grounded = "google_maps" in tool_info.get("tools_used", [])
        cost = calculate_cost(
            params.model,
            input_tokens,
            output_tokens,
            thinking_tokens,
            maps_grounded,
        )
        daily_cost = state._track_daily_cost(cog, message.author.id, cost)
        cog._log_cost(
            "chat",
            message.author.id,
            params.model,
            cost,
            daily_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            google_maps_grounded=maps_grounded,
        )
        if SHOW_COST_EMBEDS:
            embeds.append_pricing_embed(
                response_embeds,
                params.model,
                input_tokens,
                output_tokens,
                daily_cost,
                thinking_tokens,
                maps_grounded,
            )

        view = cog.views.get(message.author)
        main_conversation_id = conversation_wrapper.params.conversation_id
        if main_conversation_id is None:
            cog.logger.error("Conversation ID is None, cannot track message")
            return

        await state._strip_previous_view(cog, message.author)

        if response_embeds:
            try:
                reply_message = await message.reply(embeds=response_embeds, view=view)
            except Exception as embed_error:
                cog.logger.warning("Embed failed, sending as text: %s", embed_error)
                safe_response_text = response_text or "No response text available"
                reply_message = await message.reply(
                    content=(
                        f"**Response:**\n{safe_response_text[:1900]}"
                        f"{'...' if len(safe_response_text) > 1900 else ''}"
                    ),
                    view=view,
                )
            cog.message_to_conversation_id[reply_message.id] = main_conversation_id
            cog.last_view_messages[message.author] = reply_message
        else:
            await message.reply(
                content="An error occurred: No content to send.",
                view=view,
            )

    except Exception as error:
        description = str(error)
        cog.logger.error(
            "Error in handle_new_message_in_conversation: %s",
            description,
            exc_info=True,
        )
        if len(description) > 4000:
            description = description[:4000] + "\n\n... (error message truncated)"
        await message.reply(embed=embeds.build_error_embed(description))
        await cache._delete_conversation_cache(cog, conversation_wrapper.params)
        await attachments._cleanup_uploaded_files(cog, conversation_wrapper.params)
        conv_id = conversation_wrapper.params.conversation_id
        if conv_id is not None:
            cog.conversations.pop(conv_id, None)
        await state._cleanup_conversation(cog, message.author)
    finally:
        if typing_task:
            typing_task.cancel()


async def handle_on_message(cog: "GeminiCog", message: Any) -> None:
    """Handle the Discord on_message event for active Gemini conversations."""

    if message.author == cog.bot.user:
        return

    cog.logger.debug(
        "Received message from %s in channel %s: %r",
        message.author,
        message.channel.id,
        message.content,
    )

    for conversation_wrapper in cog.conversations.values():
        if message.channel.id != conversation_wrapper.params.channel_id:
            continue
        if message.author != conversation_wrapper.params.conversation_starter:
            continue

        cog.logger.info(
            "Processing followup message for conversation %s",
            conversation_wrapper.params.conversation_id,
        )
        await handle_new_message_in_conversation(cog, message, conversation_wrapper)
        break

    cog.logger.debug("No matching conversations found for this message")


async def chat_command(
    cog: "GeminiCog",
    ctx: ApplicationContext,
    prompt: str,
    model: str,
    system_instruction: str | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    seed: int | None,
    attachment: Attachment | None,
    url: str | None,
    temperature: float | None,
    top_p: float | None,
    media_resolution: str | None,
    thinking_level: str | None,
    thinking_budget: int | None,
    google_search: bool,
    code_execution: bool,
    google_maps: bool,
    url_context: bool,
    file_search: bool,
    custom_functions: bool,
) -> None:
    """Run the `/gemini chat` command."""

    await ctx.defer()
    typing_task = None
    channel = ctx.channel
    channel_id = getattr(channel, "id", None)
    if channel is None or channel_id is None:
        await ctx.send_followup(
            embed=embeds.build_error_embed("Unable to determine the channel for this conversation.")
        )
        return

    for conversation_wrapper in cog.conversations.values():
        if (
            conversation_wrapper.params.conversation_starter == ctx.author
            and conversation_wrapper.params.channel_id == channel_id
        ):
            await ctx.send_followup(
                embed=Embed(
                    title="Error",
                    description=(
                        "You already have an active conversation in this channel. "
                        "Please finish it before starting a new one."
                    ),
                    color=Colour.red(),
                )
            )
            return

    try:
        typing_task = asyncio.create_task(keep_typing(cog, channel))

        parts: list[dict[str, Any]] = []
        uploaded_file_names: list[str] = []
        if attachment:
            validation_error = attachments._validate_attachment_size(attachment)
            if validation_error:
                await ctx.send_followup(embed=embeds.build_error_embed(validation_error))
                if typing_task:
                    typing_task.cancel()
                return

            attachment_part = await attachments._prepare_attachment_part(
                cog,
                attachment,
                uploaded_file_names,
            )
            if attachment_part is not None:
                parts.append(attachment_part)

        if url:
            parts.append(
                {
                    "file_data": {
                        "file_uri": url,
                        "mime_type": attachments._guess_url_mime_type(url),
                    }
                }
            )

        parts.append({"text": prompt})

        selected_tool_names = {
            "google_search": google_search,
            "code_execution": code_execution,
            "google_maps": google_maps,
            "url_context": url_context,
            "file_search": file_search,
        }

        enabled_names = {name for name, enabled in selected_tool_names.items() if enabled}
        exclusive_error = check_mutually_exclusive_tools(enabled_names)
        if exclusive_error:
            await ctx.send_followup(embed=embeds.build_error_embed(exclusive_error))
            if typing_task:
                typing_task.cancel()
            return

        requested_tools = [
            deepcopy(tool_config)
            for tool in iter_tool_registry(include_custom_functions=False)
            if selected_tool_names.get(tool.canonical_id, False)
            if (tool_config := build_runtime_tool_config(tool.canonical_id)) is not None
        ]
        tools, unsupported_tools = filter_supported_tools_for_model(model, requested_tools)
        tools, incompatible_tools = filter_file_search_incompatible_tools(tools)

        enrich_error = tooling.enrich_file_search_tools(tools)
        if enrich_error:
            await ctx.send_followup(embed=embeds.build_error_embed(enrich_error))
            if typing_task:
                typing_task.cancel()
            return

        config_args: dict[str, Any] = {}
        if system_instruction is not None:
            config_args["system_instruction"] = system_instruction
        if frequency_penalty is not None:
            config_args["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            config_args["presence_penalty"] = presence_penalty
        if seed is not None:
            config_args["seed"] = seed
        if temperature is not None:
            config_args["temperature"] = temperature
        if top_p is not None:
            config_args["top_p"] = top_p
        if media_resolution is not None:
            config_args["media_resolution"] = media_resolution
        thinking_config = responses._build_thinking_config(thinking_level, thinking_budget)
        if thinking_config is not None:
            config_args["thinking_config"] = thinking_config
        if tools:
            config_args["tools"] = tools

        custom_functions_enabled = custom_functions and ENABLE_CUSTOM_TOOLS
        combination_error = validate_builtin_custom_tool_combination(
            model,
            tools,
            custom_functions_enabled,
        )
        if combination_error:
            await ctx.send_followup(embed=embeds.build_error_embed(combination_error))
            if typing_task:
                typing_task.cancel()
            return

        _add_custom_function_tools(config_args, model, custom_functions_enabled)
        _configure_tool_context_circulation(config_args, model, custom_functions_enabled)

        generation_config = types.GenerateContentConfig(**config_args) if config_args else None

        formatted_parts = []
        for part in parts:
            if "text" in part:
                formatted_parts.append(types.Part(text=part["text"]))
            elif "inline_data" in part:
                formatted_parts.append(types.Part(inline_data=part["inline_data"]))
            elif "file_data" in part:
                formatted_parts.append(
                    types.Part.from_uri(
                        file_uri=part["file_data"]["file_uri"],
                        mime_type=part["file_data"]["mime_type"],
                    )
                )

        initial_contents = [{"role": "user", "parts": formatted_parts}]
        result = await _run_agentic_loop(cog, model, initial_contents, generation_config)
        response = result.response
        response_text = response.text
        tool_info = responses.extract_tool_info(response)
        for tool_name in result.tool_calls_made:
            if tool_name not in tool_info["tools_used"]:
                tool_info["tools_used"].append(tool_name)

        truncated_prompt = truncate_text(prompt, 2000)
        description = f"**Prompt:** {truncated_prompt}\n"
        description += f"**Model:** {model}\n"
        description += (
            f"**System Instruction:** {system_instruction}\n" if system_instruction else ""
        )
        description += f"**Frequency Penalty:** {frequency_penalty}\n" if frequency_penalty else ""
        description += f"**Presence Penalty:** {presence_penalty}\n" if presence_penalty else ""
        description += f"**Seed:** {seed}\n" if seed else ""
        if temperature is not None:
            description += f"**Temperature:** {temperature}"
            if model.startswith("gemini-3") and temperature != 1.0:
                description += " (warning: Gemini 3 recommends 1.0; lower values may cause looping)"
            description += "\n"
        description += f"**Nucleus Sampling:** {top_p}\n" if top_p else ""
        description += f"**Media Resolution:** {media_resolution}\n" if media_resolution else ""
        description += (
            f"**Thinking Level:** {thinking_level.capitalize()}\n" if thinking_level else ""
        )
        description += (
            f"**Thinking Budget:** {thinking_budget}\n" if thinking_budget is not None else ""
        )
        if tools:
            active_tool_labels = [
                resolve_tool_name(tool_config) or "unknown" for tool_config in tools
            ]
            description += f"**Tools:** {', '.join(active_tool_labels)}\n"
            if unsupported_tools:
                description += (
                    "**Tools Skipped (model unsupported):** "
                    f"{', '.join(sorted(set(unsupported_tools)))}\n"
                )
            if incompatible_tools:
                description += (
                    "**Tools Skipped (incompatible with file_search):** "
                    f"{', '.join(sorted(set(incompatible_tools)))}\n"
                )

        response_embeds = [
            Embed(
                title="Conversation Started",
                description=description,
                color=Colour.green(),
            )
        ]
        thinking_text = responses.extract_thinking_text(response)
        embeds.append_thinking_embeds(response_embeds, thinking_text)
        embed_count_before_response = len(response_embeds)
        embeds.append_response_embeds(response_embeds, response_text)
        has_response = len(response_embeds) > embed_count_before_response
        embeds.append_sources_embed(response_embeds, tool_info)

        input_tokens = result.total_input_tokens
        output_tokens = result.total_output_tokens
        thinking_tokens = result.total_thinking_tokens
        maps_grounded = "google_maps" in tool_info.get("tools_used", [])
        cost = calculate_cost(model, input_tokens, output_tokens, thinking_tokens, maps_grounded)
        daily_cost = state._track_daily_cost(cog, ctx.author.id, cost)
        cog._log_cost(
            "chat",
            ctx.author.id,
            model,
            cost,
            daily_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            google_maps_grounded=maps_grounded,
        )
        if SHOW_COST_EMBEDS:
            embeds.append_pricing_embed(
                response_embeds,
                model,
                input_tokens,
                output_tokens,
                daily_cost,
                thinking_tokens,
                maps_grounded,
            )

        if not has_response:
            await ctx.send_followup("No response generated.")
            return

        interaction = ctx.interaction
        if interaction is None:
            await ctx.send_followup(
                embed=embeds.build_error_embed(
                    "Unable to determine interaction context for this conversation."
                )
            )
            return

        main_conversation_id = interaction.id
        view = ButtonView(
            conversation_starter=ctx.author,
            conversation_id=main_conversation_id,
            initial_tools=tools,
            custom_functions_enabled=custom_functions_enabled,
            get_conversation=lambda cid: state.get_conversation(cog, cid),
            on_regenerate=lambda message, conversation: handle_new_message_in_conversation(
                cog,
                message,
                conversation,
            ),
            on_stop=lambda conversation_id, user: state.end_conversation(
                cog, conversation_id, user
            ),
            on_tools_changed=lambda selected, custom, conversation: tooling._resolve_tools_for_view(
                cog,
                selected,
                custom,
                conversation,
            ),
        )
        cog.views[ctx.author] = view

        await state._strip_previous_view(cog, ctx.author)

        message = await ctx.send_followup(embeds=response_embeds, view=view)
        cog.message_to_conversation_id[message.id] = main_conversation_id
        cog.last_view_messages[ctx.author] = message

        params = ChatCompletionParameters(
            model=model,
            system_instruction=system_instruction,
            conversation_starter=ctx.author,
            channel_id=channel_id,
            conversation_id=main_conversation_id,
            temperature=temperature,
            top_p=top_p,
            media_resolution=media_resolution,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            tools=tools,
            uploaded_file_names=uploaded_file_names,
            custom_functions_enabled=custom_functions_enabled,
        )
        response_parts = responses._get_response_content_parts(response)
        history = [{"role": "user", "parts": parts}]
        for entry in initial_contents[1:]:
            history.append(entry)
        history.append({"role": "model", "parts": response_parts or [{"text": response_text}]})
        cog.conversations[main_conversation_id] = Conversation(
            params=params,
            history=history,
        )

    except Exception as error:
        await cog._send_error_followup(ctx, error, "chat")
        await state._cleanup_conversation(cog, ctx.author)
    finally:
        if typing_task:
            typing_task.cancel()


__all__ = [
    "_run_agentic_loop",
    "chat_command",
    "handle_new_message_in_conversation",
    "handle_on_message",
    "keep_typing",
]
