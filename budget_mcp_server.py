from fastmcp import FastMCP

server = FastMCP("budget-tools")

@server.tool()
def estimate_budget(destination: str, days: int) -> float:
    """Estimate travel budget in USD."""
    base_cost = 100
    return base_cost * days

if __name__ == "__main__":
    # Run with SSE transport on port 3333
    server.run(transport="sse", host="localhost", port=3333)