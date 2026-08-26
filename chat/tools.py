"""
Chat tools -- thin wrappers exposing existing anllms functionality
(feed library search, requirements report) as LLM tool-use tools.

NO new calculation logic lives here. Every function just calls something
already built and tested elsewhere in this repo, then reshapes the result
into a JSON-friendly dict for the LLM to read and relay to the user.

TOOL_DEFINITIONS below stays in Anthropic's flat schema (name /
description / input_schema) -- that's still the source of truth for
each tool's shape. Since the migration to the LiteLLM proxy,
chat/server.py reshapes this into OpenAI's nested {"type": "function",
"function": {...}} format at request time via _to_openai_tools(), so
this file doesn't need to know or care which wire format the model
provider expects.

SCOPE (matches the agreed v1 decision, since expanded): lactating dairy
cows only. Covers DMI, energy (NEL), protein (MP), all 13 NASEM
minerals, vitamins A/D/E, and water. Dry cows, heifers, and other
species are still not covered -- the system prompt instructs the
assistant to say so rather than guess.
"""

from __future__ import annotations

from anllms.feed_library.ingredient import search_feed_library
from anllms.feed_library.ration import Ration
from anllms.simulation.animal_state import AnimalState, MilkTarget
from anllms.simulation.requirements_report import build_requirements_report

TOOL_DEFINITIONS = [
    {
        "name": "search_feed_ingredient",
        "description": (
            "Search the real NASEM feed ingredient library for names matching a "
            "query (e.g. 'corn silage', 'soybean meal'). Use this to find the "
            "EXACT ingredient name before calling calculate_lactating_cow_requirements "
            "-- ingredient names must match the library exactly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text, e.g. 'corn silage'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_lactating_cow_requirements",
        "description": (
            "Calculate dry matter intake, energy (NEL), protein (MP), all "
            "13 NASEM minerals, vitamins A/D/E, and water requirement for a "
            "LACTATING dairy cow fed a specific ration. Only valid for "
            "lactating cows -- do not use for dry cows, heifers, or other "
            "species. All ration ingredient names must be exact matches "
            "from search_feed_ingredient."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bw_kg": {"type": "number", "description": "Body weight, kg"},
                "bcs": {"type": "number", "description": "Body condition score, 1-5 scale"},
                "days_in_milk": {"type": "integer", "description": "Days in milk (DIM)"},
                "parity": {"type": "integer", "description": "1 = first lactation, 2+ = multiparous"},
                "milk_yield_kg": {"type": "number", "description": "Milk yield, kg/day"},
                "milk_fat_pct": {"type": "number", "description": "Milk fat, %"},
                "milk_true_protein_pct": {"type": "number", "description": "Milk true protein, %"},
                "milk_lactose_pct": {"type": "number", "description": "Milk lactose, %"},
                "ration_items": {
                    "type": "array",
                    "description": (
                        "List of ingredients and their inclusion rates. OPTIONAL -- "
                        "if omitted, a standard reference diet (nasem_dairy's own "
                        "built-in demo ration) is used as a placeholder, and the "
                        "result will say so in its warnings."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Exact feed library name"},
                            "kg_dm_per_day": {"type": "number", "description": "kg dry matter per day"},
                        },
                        "required": ["name", "kg_dm_per_day"],
                    },
                },
            },
            "required": [
                "bw_kg", "bcs", "days_in_milk", "parity", "milk_yield_kg",
                "milk_fat_pct", "milk_true_protein_pct", "milk_lactose_pct",
            ],
        },
    },
    {
        "name": "explain_component",
        "description": (
            "Get the full citation, assumptions, limitations, and reasoning "
            "behind one specific number from the most recent requirements "
            "calculation in this conversation. Use this when the user asks "
            "'why' or 'where does that come from' about a specific value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "enum": [
                        "dmi", "nel_maintenance", "nel_lactation", "nel_supply_total",
                        "mp_maintenance", "mp_lactation", "mp_supply_total",
                        "water",
                        "mineral_Ca", "mineral_P", "mineral_Mg", "mineral_Na",
                        "mineral_Cl", "mineral_K", "mineral_S", "mineral_Co",
                        "mineral_Cu", "mineral_Fe", "mineral_Mn", "mineral_Se",
                        "mineral_Zn", "mineral_I",
                        "vitamin_A", "vitamin_D", "vitamin_E",
                    ],
                    "description": (
                        "Which value to explain. For minerals/vitamins, use "
                        "the 'mineral_<Symbol>' or 'vitamin_<Symbol>' form, "
                        "e.g. 'mineral_Ca' for calcium, 'vitamin_E' for vitamin E."
                    ),
                }
            },
            "required": ["component"],
        },
    },
]


class ChatSession:
    """
    Holds the most recent RequirementsReport so explain_component can be
    called in a later turn without recomputing anything. One session =
    one browser tab's conversation; not designed for multi-user
    deployment as-is.
    """

    def __init__(self):
        self.last_report = None

    def dispatch(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name == "search_feed_ingredient":
            return {"matches": search_feed_library(tool_input["query"])}

        if tool_name == "calculate_lactating_cow_requirements":
            return self._calculate_requirements(tool_input)

        if tool_name == "explain_component":
            return self._explain_component(tool_input["component"])

        return {"error": f"Unknown tool: {tool_name}"}

    def _calculate_requirements(self, args: dict) -> dict:
        animal = AnimalState(
            bw_kg=args["bw_kg"], bcs=args["bcs"],
            days_in_milk=args["days_in_milk"], parity=args["parity"],
        )
        milk = MilkTarget(
            yield_kg=args["milk_yield_kg"], fat_pct=args["milk_fat_pct"],
            true_protein_pct=args["milk_true_protein_pct"],
            lactose_pct=args["milk_lactose_pct"],
        )
        ration_items = args.get("ration_items") or []
        used_default_diet = not ration_items
        if ration_items:
            ration = Ration()
            for item in ration_items:
                ration.add(item["name"], item["kg_dm_per_day"])
        else:
            ration = Ration.guelph_base_diet()

        missing = ration.validate_feedstuffs_exist()
        if missing:
            return {
                "error": (
                    f"These ingredient names were not found in the feed library: "
                    f"{missing}. Use search_feed_ingredient to find exact names."
                )
            }

        try:
            report = build_requirements_report(animal, milk, ration)
        except Exception as e:
            return {"error": f"Calculation failed: {e}"}

        if used_default_diet:
            report.warnings.insert(
                0,
                "No diet was specified, so this used nasem_dairy's own "
                "built-in demo ration (alfalfa meal, canola meal, corn "
                "silage, corn grain HM -- ~24.5 kg DM/d) as a placeholder, "
                "not a diet formulated for this animal. Anything diet-"
                "dependent (DMI via Eq. 2-2, MP/mineral/vitamin supply) "
                "reflects this placeholder, not a real ration.",
            )

        self.last_report = report

        return {
            "dmi_kg_per_day": round(report.dmi_result.value, 2),
            "dmi_equation_used": report.dmi_equation_used,
            "nel_requirement_mcal_per_day": round(report.total_nel_requirement_mcal, 2),
            "nel_supply_mcal_per_day": round(report.nel_supply_total.value, 2),
            "nel_balance_mcal_per_day": round(report.nel_balance_mcal, 2),
            "mp_requirement_g_per_day": round(report.total_mp_requirement_g, 1),
            "mp_supply_g_per_day": round(report.mp_supply_total.value, 1),
            "mp_balance_g_per_day": round(report.mp_balance_g, 1),
            "water_requirement_kg_per_day": round(report.water_result.value, 1),
            "minerals": {
                symbol: {
                    "requirement": round(result.value, 3),
                    "unit": result.unit,
                    "balance": round(report.mineral_balances[symbol], 3)
                    if symbol in report.mineral_balances else None,
                }
                for symbol, result in report.mineral_results.items()
            },
            "vitamins": {
                symbol: {
                    "requirement": round(result.value, 1),
                    "unit": result.unit,
                    "balance": round(report.vitamin_balances[symbol], 1)
                    if symbol in report.vitamin_balances else None,
                }
                for symbol, result in report.vitamin_results.items()
            },
            "warnings": report.warnings,
            "note": (
                "Mineral/vitamin 'balance' values come directly from the "
                "underlying reference model, not from independently-cited "
                "supply equations -- mention this if the user asks about "
                "mineral/vitamin supply specifically. "
                "Use explain_component if the user asks why any of these "
                "numbers are what they are (e.g. component='mineral_Ca')."
            ),
        }

    def _explain_component(self, component: str) -> dict:
        if self.last_report is None:
            return {
                "error": (
                    "No requirements calculation has been run yet in this "
                    "conversation -- call calculate_lactating_cow_requirements first."
                )
            }
        mapping = {
            "dmi": self.last_report.dmi_result,
            "nel_maintenance": self.last_report.nel_maintenance,
            "nel_lactation": self.last_report.nel_lactation,
            "nel_supply_total": self.last_report.nel_supply_total,
            "mp_maintenance": self.last_report.mp_maintenance,
            "mp_lactation": self.last_report.mp_lactation,
            "mp_supply_total": self.last_report.mp_supply_total,
            "water": self.last_report.water_result,
        }
        if component.startswith("mineral_"):
            symbol = component.removeprefix("mineral_")
            result = self.last_report.mineral_results.get(symbol)
        elif component.startswith("vitamin_"):
            symbol = component.removeprefix("vitamin_")
            result = self.last_report.vitamin_results.get(symbol)
        else:
            result = mapping.get(component)

        if result is None:
            return {"error": f"Unknown component: {component}"}
        return {"explanation": result.explain()}
