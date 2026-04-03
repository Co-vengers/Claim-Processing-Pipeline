from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional

from agents.segregator import run_segregator
from agents.id_agent import run_id_agent
from agents.discharge_agent import run_discharge_agent
from agents.bill_agent import run_bill_agent

class ClaimState(TypedDict):
    claim_id: str
    pages: list[dict]

    classified_pages: dict

    identity_data: dict
    discharge_data: dict
    bill_data: dict

    final_result: Optional[dict]

def segregator_node(state: ClaimState) -> dict:
    """Classifies each page into a document type."""
    print("\n===== [Node: Segregator] =====")
    classified = run_segregator(state["pages"])
    return {"classified_pages": classified}


def id_agent_node(state: ClaimState) -> dict:
    """Extracts identity info from identity_document pages."""
    print("\n===== [Node: ID Agent] =====")
    pages = state["classified_pages"].get("identity_document", [])
    result = run_id_agent(pages)
    return {"identity_data": result}


def discharge_agent_node(state: ClaimState) -> dict:
    """Extracts hospital/diagnosis info from discharge_summary pages."""
    print("\n===== [Node: Discharge Agent] =====")
    pages = state["classified_pages"].get("discharge_summary", [])
    result = run_discharge_agent(pages)
    return {"discharge_data": result}


def bill_agent_node(state: ClaimState) -> dict:
    """Extracts billing items from itemized_bill pages."""
    print("\n===== [Node: Bill Agent] =====")
    pages = state["classified_pages"].get("itemized_bill", [])
    result = run_bill_agent(pages)
    return {"bill_data": result}


def aggregator_node(state: ClaimState) -> dict:
    """Combines all agent outputs into a single final JSON."""
    print("\n===== [Node: Aggregator] =====")
    final = {
        "claim_id": state["claim_id"],
        "identity":          state.get("identity_data", {}),
        "discharge_summary": state.get("discharge_data", {}),
        "itemized_bill":     state.get("bill_data", {}),
        "document_types_found": list(state["classified_pages"].keys())
    }
    return {"final_result": final}


def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.
    Call this once at startup; reuse the compiled graph for every request.
    """

    graph = StateGraph(ClaimState)

    graph.add_node("segregator",       segregator_node)
    graph.add_node("id_agent",         id_agent_node)
    graph.add_node("discharge_agent",  discharge_agent_node)
    graph.add_node("bill_agent",       bill_agent_node)
    graph.add_node("aggregator",       aggregator_node)

    graph.add_edge(START, "segregator")

    graph.add_edge("segregator", "id_agent")
    graph.add_edge("segregator", "discharge_agent")
    graph.add_edge("segregator", "bill_agent")

    graph.add_edge("id_agent",        "aggregator")
    graph.add_edge("discharge_agent", "aggregator")
    graph.add_edge("bill_agent",      "aggregator")

    graph.add_edge("aggregator", END)

    return graph.compile()


compiled_graph = build_graph()
