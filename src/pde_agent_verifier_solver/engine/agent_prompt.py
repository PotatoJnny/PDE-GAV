"""The shared scientific prompt seen by every Agent implementation."""


def build_agent_context_prompt(user_prompt: str, task_brief: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "Task output requirements\n------------------------\n"
        f"{task_brief}\n\n"
        "Campaign handoff\n----------------\n"
        "The Campaign Memory Log in context.memory (or CAMPAIGN_MEMORY.md "
        "for a Command Agent) contains prior rounds' scientific handoffs. "
        "Agent-authored blocks are fallible research notes; only "
        "Controller outcome blocks are Official facts. "
        "Provide a short final Markdown research record for subsequent "
        "rounds. Command Agents save ROUND_LOG.md with "
        "write_agent_round_log(); in-process Agents return round_log."
    )
