"""
Live rate fetching from Aave and Lido.

Fetches current rates for stETH-ETH looping analysis:
- ETH borrow rate from Aave
- stETH supply (lending) rate from Aave
- stETH staking rate from Lido
"""

import requests
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LiveRates:
    """Container for live rate data."""
    eth_borrow_rate: float      # Aave ETH borrow APR
    steth_supply_rate: float    # Aave stETH supply/lending APR
    steth_staking_rate: float   # Lido stETH staking APR
    chain: str                  # Chain name
    timestamp: Optional[str] = None

    @property
    def r_borrow(self) -> float:
        """ETH borrow rate."""
        return self.eth_borrow_rate

    @property
    def r_steth_lend(self) -> float:
        """stETH lending rate on Aave."""
        return self.steth_supply_rate

    @property
    def r_steth_stake(self) -> float:
        """stETH staking rate from Lido."""
        return self.steth_staking_rate

    @property
    def r_stake(self) -> float:
        """Total stETH yield = staking + lending."""
        return self.steth_staking_rate + self.steth_supply_rate

    @property
    def raw_spread(self) -> float:
        """Raw spread = r_stake - r_borrow."""
        return self.r_stake - self.r_borrow


# Aave V3 subgraph endpoints by chain
AAVE_SUBGRAPHS = {
    "Ethereum": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
    "Arbitrum": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-arbitrum",
    "Optimism": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-optimism",
    "Polygon": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-polygon",
    "Base": "https://api.thegraph.com/subgraphs/name/aave/protocol-v3-base",
}

# Asset addresses (lowercase)
ASSET_ADDRESSES = {
    "Ethereum": {
        "WETH": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
        "wstETH": "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",
        "stETH": "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",
    },
    "Arbitrum": {
        "WETH": "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
        "wstETH": "0x5979d7b546e38e414f7e9822514be443a4800529",
    },
    "Optimism": {
        "WETH": "0x4200000000000000000000000000000000000006",
        "wstETH": "0x1f32b1c2345538c0c6f582fcb022739c4a194ebb",
    },
}

# Lido API endpoint
LIDO_APR_API = "https://eth-api.lido.fi/v1/protocol/steth/apr/sma"


def fetch_lido_staking_rate() -> Optional[float]:
    """
    Fetch current stETH staking APR from Lido API.

    Returns:
        Staking APR as decimal (e.g., 0.035 for 3.5%), or None on error
    """
    try:
        response = requests.get(LIDO_APR_API, timeout=10)
        response.raise_for_status()
        data = response.json()
        # API returns APR as percentage, convert to decimal
        apr = float(data.get("data", {}).get("smaApr", 0))
        return apr / 100.0
    except Exception as e:
        print(f"Error fetching Lido rate: {e}")
        return None


def fetch_aave_rates(chain: str = "Ethereum") -> Optional[Dict[str, float]]:
    """
    Fetch current Aave V3 rates for ETH and stETH.

    Args:
        chain: Chain name (Ethereum, Arbitrum, Optimism, etc.)

    Returns:
        Dict with 'eth_borrow_rate' and 'steth_supply_rate' as decimals, or None on error
    """
    subgraph_url = AAVE_SUBGRAPHS.get(chain)
    if not subgraph_url:
        print(f"Unknown chain: {chain}")
        return None

    addresses = ASSET_ADDRESSES.get(chain, {})
    weth_addr = addresses.get("WETH", "").lower()
    wsteth_addr = addresses.get("wstETH", "").lower()

    if not weth_addr or not wsteth_addr:
        print(f"Asset addresses not configured for {chain}")
        return None

    # GraphQL query for reserve data
    query = """
    {
      reserves(where: {underlyingAsset_in: ["%s", "%s"]}) {
        underlyingAsset
        symbol
        liquidityRate
        variableBorrowRate
      }
    }
    """ % (weth_addr, wsteth_addr)

    try:
        response = requests.post(
            subgraph_url,
            json={"query": query},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        reserves = data.get("data", {}).get("reserves", [])

        eth_borrow_rate = None
        steth_supply_rate = None

        for reserve in reserves:
            asset = reserve.get("underlyingAsset", "").lower()
            symbol = reserve.get("symbol", "")

            # Rates are in RAY (27 decimals), convert to decimal APR
            if asset == weth_addr or "ETH" in symbol.upper():
                # Variable borrow rate
                rate_ray = int(reserve.get("variableBorrowRate", 0))
                eth_borrow_rate = rate_ray / 1e27

            if asset == wsteth_addr or "STETH" in symbol.upper():
                # Supply/liquidity rate
                rate_ray = int(reserve.get("liquidityRate", 0))
                steth_supply_rate = rate_ray / 1e27

        if eth_borrow_rate is not None and steth_supply_rate is not None:
            return {
                "eth_borrow_rate": eth_borrow_rate,
                "steth_supply_rate": steth_supply_rate,
            }

    except Exception as e:
        print(f"Error fetching Aave rates: {e}")

    return None


def fetch_aave_rates_defi_llama(chain: str = "Ethereum") -> Optional[Dict[str, float]]:
    """
    Fetch Aave rates using DeFi Llama API.

    Uses both the pools endpoint (for supply rates) and lendBorrow endpoint (for borrow rates).

    Args:
        chain: Chain name

    Returns:
        Dict with rates or None on error
    """
    try:
        # Fetch pools data (supply rates)
        pools_response = requests.get(
            "https://yields.llama.fi/pools",
            timeout=15
        )
        pools_response.raise_for_status()
        pools = pools_response.json().get("data", [])

        # Fetch lendBorrow data (borrow rates)
        borrow_response = requests.get(
            "https://yields.llama.fi/lendBorrow",
            timeout=15
        )
        borrow_response.raise_for_status()
        borrow_data = borrow_response.json()

        # Create borrow rate lookup by pool ID
        borrow_map = {b["pool"]: b for b in borrow_data}

        eth_borrow_rate = None
        steth_supply_rate = None

        chain_lower = chain.lower()

        for pool in pools:
            project = pool.get("project", "").lower()
            pool_chain = pool.get("chain", "").lower()
            symbol = pool.get("symbol", "").upper()

            # Only consider Aave V3 on the specified chain
            if project != "aave-v3":
                continue
            if pool_chain != chain_lower:
                continue

            pool_id = pool.get("pool")
            borrow_info = borrow_map.get(pool_id, {})

            # Get ETH borrow rate
            if symbol in ["WETH", "ETH"]:
                borrow_apy = borrow_info.get("apyBaseBorrow")
                if borrow_apy is not None:
                    # DeFi Llama returns percentage, convert to decimal
                    eth_borrow_rate = borrow_apy / 100.0

            # Get stETH/wstETH supply rate
            if "STETH" in symbol:
                supply_apy = pool.get("apyBase")
                if supply_apy is not None:
                    # DeFi Llama returns percentage, convert to decimal
                    steth_supply_rate = supply_apy / 100.0

        if eth_borrow_rate is not None and steth_supply_rate is not None:
            return {
                "eth_borrow_rate": eth_borrow_rate,
                "steth_supply_rate": steth_supply_rate,
            }

    except Exception as e:
        print(f"Error fetching from DeFi Llama: {e}")

    return None


def fetch_live_rates(chain: str = "Ethereum") -> Optional[LiveRates]:
    """
    Fetch all live rates for stETH-ETH looping.

    Args:
        chain: Chain name for Aave rates

    Returns:
        LiveRates object or None on error
    """
    # Fetch Lido staking rate
    steth_staking_rate = fetch_lido_staking_rate()
    if steth_staking_rate is None:
        steth_staking_rate = 0.035  # Fallback ~3.5%

    # Use DeFi Llama API (Aave subgraph is deprecated)
    aave_rates = fetch_aave_rates_defi_llama(chain)

    if aave_rates is None:
        return None

    return LiveRates(
        eth_borrow_rate=aave_rates["eth_borrow_rate"],
        steth_supply_rate=aave_rates["steth_supply_rate"],
        steth_staking_rate=steth_staking_rate,
        chain=chain,
    )


# Available chains for selection
AVAILABLE_CHAINS = list(AAVE_SUBGRAPHS.keys())
