import os
import textwrap
import datetime
from dirigera.devices.outlet import Outlet
from dirigera.devices.controller import Controller
from dirigera.devices.light import Light
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from dirigera.hub.hub import Hub

###############################################################################
### Program variables: Update in accordance with workshop instructions!     ###
###############################################################################

# Change to run the agent after receiving a command
enable_agent = False

# Change to allow the agent to access the IKEA devices
enable_dirigera = False


###############################################################################
### Base Data Models for describing the agent and device inputs and outputs ###
###############################################################################
class AgentRequest(BaseModel):
    prompt: str


class LightStatus(BaseModel):
    is_on: bool | None = Field(description="Whether the lights are on or off")
    light_level: int | None = Field(
        description="Brightness of the lights from 1 (lowest) to 100 (highest)",
        ge=0,
        le=100,
        default=100,
    )
    light_hue: float | None = Field(
        description="Hue of the lights from 0 to 360", ge=0, le=100, default=0
    )
    light_saturation: float | None = Field(
        description="Saturation of the lights from 0 to 1", ge=0, le=1, default=1
    )


########################################################################################
### Agent Dependencies: Describes the current state of the agent and its environment ###
########################################################################################


class DeviceDeps:
    def __init__(self, has_dirigera: bool = True):
        self.has_dirigera: bool = has_dirigera
        if self.has_dirigera:
            self.dirigera_hub: Hub = Hub(
                token=os.environ["DIRIGERA_HUB_TOKEN"],
                ip_address="host.docker.internal",
            )

            self.desk_lamp_outlet: Outlet = self.dirigera_hub.get_outlet_by_name(
                outlet_name="Tischlampe"
            )
            self.light_chain_outlet: Outlet = self.dirigera_hub.get_outlet_by_name(
                outlet_name="Fotolicht"
            )
            self.room_lights: Light = self.dirigera_hub.get_light_by_name(
                lamp_name="Licht 1"
            )
            self.remote_control: Controller = self.dirigera_hub.get_controller_by_name(
                controller_name="Fernbedienung"
            )

    def desk_lamp_status(self) -> str:
        if not self.has_dirigera:
            return "off"

        return "on" if self.desk_lamp_outlet.attributes.is_on else "off"

    def light_chain_status(self) -> str:
        if not self.has_dirigera:
            return "off"

        return "on" if self.light_chain_outlet.attributes.is_on else "off"

    def room_lights_status(self) -> str:
        if not self.has_dirigera:
            return "off"

        current_hue = self.room_lights.attributes.color_hue
        current_saturation = self.room_lights.attributes.color_saturation
        current_brightness = self.room_lights.attributes.light_level

        # Since this is a color-changing lamp, we expect all the color attributes to be valid
        assert current_hue is not None
        assert current_saturation is not None
        assert current_brightness is not None

        return (
            f"on, with color hue {current_hue:.1f}, saturation {current_saturation * 100:.1f}% and brightness {float(current_brightness) / 100:.1f}%."
            if self.room_lights.attributes.is_on
            else "off"
        )


########################################################################################
### Agent Configuration: Creates and calls the Agent itself                          ###
########################################################################################

all_messages: list[ModelMessage] | None = None
devices: DeviceDeps = DeviceDeps(has_dirigera=enable_dirigera)

agent = Agent(
    "openai:gpt-4o",
    deps_type=DeviceDeps,
    system_prompt=textwrap.dedent(
        """
            You are a smart-home assistant capable of controlling various devices.
            Please keep all your responses short!
        """
    ),
)


async def run_agent(req: AgentRequest) -> str:
    if not enable_agent:
        return f"Echo: {req.prompt}"
    else:
        global all_messages
        result = await agent.run(req.prompt, deps=devices, message_history=all_messages)
        all_messages = result.all_messages()
        return result.output


async def reset_agent() -> None:
    global all_messages
    all_messages = None


########################################################################################
### Agent Instructions and Tools : Used by the agent to interact with the world      ###
########################################################################################


@agent.instructions
def get_available_devices(ctx: RunContext[DeviceDeps]) -> str:
    return textwrap.dedent(
        f"""
        You currently have access to the following devices:
        - A small desk lamp, controlled through the tool "toggle_desk_light".
          This desk lamp is currently {ctx.deps.desk_lamp_status()}.
        - A light chain, controlled through the tool "toggle_light_chain".
          This light chain is currently {ctx.deps.light_chain_status()}.
        - A set of room lights, controlled through the tool "toggle_room_lights"
          The room lights are currently {ctx.deps.room_lights_status()}.

        Make use of these tools to control the smart-home devices and answer the user's queries.

        In addition to the device controls, you have access to a tool "get_current_time"
        for accessing the current date and time.
        """
    )


@agent.tool
def toggle_desk_light(ctx: RunContext[DeviceDeps], is_on: bool) -> None:
    """Turn the desk light on or off

    Args:
        is_on: True to turn on the lamp, False to turn it off.
    """
    if ctx.deps.has_dirigera:
        ctx.deps.desk_lamp_outlet.set_on(is_on)


@agent.tool
def toggle_light_chain(ctx: RunContext[DeviceDeps], is_on: bool) -> None:
    """Turn the light chain on or off

    Args:
        is_on: True to turn on the lamp, False to turn it off.
    """
    if ctx.deps.has_dirigera:
        ctx.deps.light_chain_outlet.set_on(is_on)


@agent.tool
def toggle_room_lights(
    ctx: RunContext[DeviceDeps], new_status: LightStatus
) -> LightStatus:
    """Change color and brightness of the room lights

    Args:
        new_status: Desired status of the room lights.
    """
    if ctx.deps.has_dirigera:
        if new_status.is_on is not None:
            ctx.deps.room_lights.set_light(new_status.is_on)

        if new_status.light_level is not None:
            ctx.deps.room_lights.set_light_level(new_status.light_level)

        if new_status.light_hue is not None and new_status.light_saturation is not None:
            ctx.deps.room_lights.set_light_color(
                new_status.light_hue, new_status.light_saturation
            )

        new_attributes = ctx.deps.room_lights.attributes

        return LightStatus(
            is_on=new_attributes.is_on,
            light_level=new_attributes.light_level,
            light_hue=new_attributes.color_hue,
            light_saturation=new_attributes.color_saturation,
        )
    else:
        return LightStatus(is_on=False)


@agent.tool
def get_current_time(ctx: RunContext[DeviceDeps]) -> str:
    """Get the current date and time."""
    # Get current date and time
    now = datetime.datetime.now()

    # Extract parts
    weekday = now.strftime("%A")
    day = now.day
    month = now.strftime("%B")
    year = now.year
    hour = now.strftime("%I")
    minute = now.strftime("%M")
    am_pm = now.strftime("%p")

    # Format output
    return f"Today is {weekday}, {day}. {month} {year}. It is currently {hour}:{minute}{am_pm}."
