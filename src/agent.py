from dirigera.devices.outlet import Outlet
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dirigera import Hub
import os
import textwrap

has_dirigera = True

if has_dirigera:
    dirigera_hub = Hub(
            token=os.environ["DIRIGERA_HUB_TOKEN"],
            ip_address="host.docker.internal"
        )

    desk_lamp_outlet: Outlet = dirigera_hub.get_outlet_by_name(outlet_name="Tischlampe")
    light_chain_outlet: Outlet = dirigera_hub.get_outlet_by_name(outlet_name="Fotolicht")
    room_lights = dirigera_hub.get_light_by_name(lamp_name="Licht 1")

agent: Agent[None, str] = Agent(
    'openai:gpt-4o',
    system_prompt=textwrap.dedent(
        """
            You are a smart-home assistant capable of controlling various devices.
        """
    ))

def get_agent():
    return agent


class AgentRequest(BaseModel):
    prompt: str

class LightStatus(BaseModel):
    is_on: bool | None = Field(description="Whether the lights are on or off")
    light_level: int | None = Field(description="Brightness of the lights from 1 (lowest) to 100 (highest)", ge=0, le=100, default=100)
    light_hue: float | None= Field(description="Hue of the lights from 0 to 360", ge=0, le=100, default=0)
    light_saturation: float | None = Field(description="Saturation of the lights from 0 to 1", ge=0, le=1, default=1)


@agent.system_prompt
def get_available_devices(ctx: RunContext[None]) -> str:
    return textwrap.dedent(
        """
        You currently have access to the following devices:
        - A small desk lamp, controlled through the tool "toggle_desk_light".
        - A light chain, controlled through the tool "toggle_light_chain".
        - A set of room lights, controlled through the tool "toggle_room_lights"

        Make use of these tools to control the smart-home devices and answer the user's queries.
        """
    )

@agent.tool
def toggle_desk_light(ctx: RunContext[None], is_on: bool) -> None:
    """Turn the desk light on or off
    
    Args:
        is_on: True to turn on the lamp, False to turn it off.
    """
    if has_dirigera:
        desk_lamp_outlet.set_on(is_on)

@agent.tool
def toggle_light_chain(ctx: RunContext[None], is_on: bool) -> None:
    """Turn the light chain on or off
    
    Args:
        is_on: True to turn on the lamp, False to turn it off.
    """
    if has_dirigera:
        light_chain_outlet.set_on(is_on)

@agent.tool
def toggle_room_lights(ctx: RunContext[None], new_status: LightStatus) -> LightStatus:
    """Change color and brightness of the room lights

    Args:
        new_status: Desired status of the room lights.
    """
    if has_dirigera:
        if new_status.is_on is not None:
            room_lights.set_light(new_status.is_on)
        
        if new_status.light_level is not None:
            room_lights.set_light_level(new_status.light_level)

        if new_status.light_hue is not None and new_status.light_saturation is not None:
            room_lights.set_light_color(new_status.light_hue, new_status.light_saturation)

        new_attributes = room_lights.attributes

        return LightStatus(
            is_on=new_attributes.is_on,
            light_level = new_attributes.light_level,
            light_hue=new_attributes.color_hue,
            light_saturation=new_attributes.color_saturation
        )
    else:
        return LightStatus(is_on=False)