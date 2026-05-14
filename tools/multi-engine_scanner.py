"""
Multi-Engine IOC Scanner with LangChain Multi-Agent Architecture.

Scans Indicators of Compromise (IPs, domains, URLs, hashes, emails) across
multiple threat-intelligence engines using a supervisor-worker agent pattern.
Each engine is a LangChain BaseTool; a coordinating agent (deterministic or
LLM-driven) routes the IOC to the right scanners and aggregates results.

Architecture:
    ┌──────────────────────┐
    │   MultiAgentScanner   │  ← orchestrator / supervisor
    │  (routes IOC → tools) │
    └──────┬───────────────┘
           │
    ┌──────┴──────────────────────────┐
    │         Scanner Tools            │
    │  ┌──────────┐ ┌───────────────┐  │
    │  │VirusTotal│ │MalwareBazaar  │  │
    │  └──────────┘ └───────────────┘  │
    │  ┌──────────┐ ┌───────────────┐  │
    │  │AbuseIPDB │ │MITRE ATT&CK   │  │
    │  └──────────┘ └───────────────┘  │
    └──────────────────────────────────┘
           │
    ┌──────┴──────┐
    │  ScanReport  │  ← list of ScanFinding
    └─────────────┘

Usage:
    from multi_engine_scanner import scan_ioc

    report = scan_ioc("ip", "8.8.8.8")
    print(report.summary)
    print(report.to_list())          # list of dicts — "list of information"

    # LLM-driven agent mode:
    from multi_engine_scanner import MultiAgentScanner
    scanner = MultiAgentScanner(use_llm=True)
    output = scanner.scan_with_agent("hash", "<sha256>")

CLI:
    python multi-engine_scanner.py ip 8.8.8.8
    python multi-engine_scanner.py hash 2b8ba28ca8ce0c5d3b6f25b0e4d2e8a6

API keys (environment variables):
    VIRUSTOTAL_API_KEY   — VirusTotal v3 API
    ABUSEIPDB_API_KEY    — AbuseIPDB v2 API
    OPENAI_API_KEY       — LLM agent mode (or OPENAI_BASE_URL for local)
    MalwareBazaar and MITRE ATT&CK need no keys.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pydantic
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------
import httpx

# ---------------------------------------------------------------------------
# LangChain — core (optional but strongly recommended)
# ---------------------------------------------------------------------------
try:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.tools import BaseTool

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False
    BaseTool = object  # type: ignore[misc,assignment]
    CallbackManagerForToolRun = None  # type: ignore[misc,assignment]

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    AGENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    AGENT_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI

    CHAT_AVAILABLE = True
except ImportError:  # pragma: no cover
    CHAT_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================

MALWARE_BAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
VIRUSTOTAL_API = "https://www.virustotal.com/api/v3/"
ABUSEIPDB_API = "https://api.abuseipdb.com/api/v2/check"

# Regex patterns for IOC validation
_PATTERNS: Dict[str, re.Pattern] = {
    "ipv4": re.compile(
        r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
    ),
    "domain": re.compile(
        r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
    ),
    "md5": re.compile(r"^[a-fA-F0-9]{32}$"),
    "sha1": re.compile(r"^[a-fA-F0-9]{40}$"),
    "sha256": re.compile(r"^[a-fA-F0-9]{64}$"),
    "url": re.compile(r"^https?://"),
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
}


# ============================================================================
# Enums
# ============================================================================


class IOCType(str, Enum):
    """Supported IOC types."""

    IP = "ip"
    DOMAIN = "domain"
    URL = "url"
    HASH = "hash"
    EMAIL = "email"


class ScanEngine(str, Enum):
    """Threat-intelligence engines available for scanning."""

    MALWARE_BAZAAR = "malware_bazaar"
    VIRUSTOTAL = "virus_total"
    ABUSEIPDB = "abuse_ipdb"
    MITRE_ATTACK = "mitre_attack"


# ============================================================================
# Pydantic models
# ============================================================================


class IOCInput(BaseModel):
    """Validated IOC input."""

    ioc_type: IOCType = Field(description="Type: ip, domain, url, hash, email")
    ioc_value: str = Field(description="The IOC value to scan")

    @field_validator("ioc_value")
    @classmethod
    def validate_ioc(cls, v: str, info: Any) -> str:
        """Ensure the value matches its declared IOC type."""
        ioc_type = info.data.get("ioc_type") if hasattr(info, "data") else None
        if ioc_type is None:
            return v

        checks = {
            IOCType.IP: [("ipv4", _PATTERNS["ipv4"])],
            IOCType.DOMAIN: [("domain", _PATTERNS["domain"])],
            IOCType.HASH: [
                ("md5", _PATTERNS["md5"]),
                ("sha1", _PATTERNS["sha1"]),
                ("sha256", _PATTERNS["sha256"]),
            ],
            IOCType.URL: [("url", _PATTERNS["url"])],
        }
        patterns = checks.get(ioc_type, [])
        if patterns and not any(p.match(v) for _, p in patterns):
            label = "/".join(n for n, _ in patterns)
            raise ValueError(f"Invalid {ioc_type.value} (expected {label}): {v!r}")
        return v


class ScanFinding(BaseModel):
    """A single finding from one engine."""

    engine: ScanEngine = Field(description="Engine that produced this finding")
    key: str = Field(description="Finding label")
    value: str = Field(description="Finding value")
    severity: Optional[str] = Field(
        default=None, description="low | medium | high | critical"
    )


class ScanReport(BaseModel):
    """Aggregated scan report returned to the caller."""

    ioc_type: IOCType
    ioc_value: str
    timestamp: datetime = Field(default_factory=datetime.now)
    findings: List[ScanFinding] = Field(default_factory=list)
    raw_results: Dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Human-readable summary of all findings."""
        if not self.findings:
            return f"No findings for {self.ioc_type.value}: {self.ioc_value}"
        lines = [f"Scan Report — {self.ioc_type.value}: {self.ioc_value}"]
        for f in self.findings:
            sev = f"  [{f.severity}]" if f.severity else ""
            lines.append(f"  [{f.engine.value}] {f.key}: {f.value}{sev}")
        return "\n".join(lines)

    def to_list(self) -> List[Dict[str, str]]:
        """Return findings as a plain list of dicts — the 'list of information'."""
        return [f.model_dump() for f in self.findings]


# ============================================================================
# Scanner tools  (LangChain BaseTool subclasses)
# ============================================================================


class _BaseToolMixin:
    """Shared helpers for all scanner tools (works with or without LangChain)."""

    @staticmethod
    def _safe_json(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response: {raw[:200]}"}


class MalwareBazaarTool(_BaseToolMixin, BaseTool if LANGCHAIN_AVAILABLE else object):  # type: ignore[misc]
    """Query file hashes, tags, and signatures against MalwareBazaar (abuse.ch).

    No API key required.
    """

    name: str = "malware_bazaar_scanner"
    description: str = (
        "Scan a file hash (MD5/SHA1/SHA256), tag, or signature against "
        "MalwareBazaar by abuse.ch. Input: JSON with 'query_type' "
        "(hash|tag|signature) and 'value'. "
        "Returns: JSON with malware sample metadata."
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        data = self._safe_json(query)
        qtype = data.get("query_type", "hash")
        value = data.get("value", "")

        handlers = {
            "hash": self._query_hash,
            "tag": self._query_tag,
            "signature": self._query_signature,
        }
        handler = handlers.get(qtype)
        if handler is None:
            return json.dumps({"error": f"Unknown query_type: {qtype}"})
        return handler(value)

    # ------------------------------------------------------------------
    def _query_hash(self, hash_value: str) -> str:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    MALWARE_BAZAAR_API,
                    data={"query": "get_info", "hash": hash_value},
                )
                resp.raise_for_status()
                result: Dict[str, Any] = resp.json()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        if result.get("query_status") != "ok":
            return json.dumps(
                {"found": False, "message": result.get("query_status", "error")}
            )

        data = result.get("data", [])
        if not data:
            return json.dumps({"found": False, "message": "Hash not found"})

        sample = data[0]
        return json.dumps(
            {
                "found": True,
                "sha256_hash": sample.get("sha256_hash"),
                "md5_hash": sample.get("md5_hash"),
                "sha1_hash": sample.get("sha1_hash"),
                "file_name": sample.get("file_name"),
                "file_type": sample.get("file_type"),
                "file_size": sample.get("file_size"),
                "signature": sample.get("signature"),
                "tags": sample.get("tags", []),
                "first_seen": sample.get("first_seen"),
                "last_seen": sample.get("last_seen"),
                "reporter": sample.get("reporter"),
                "comment": sample.get("comment"),
            }
        )

    def _query_tag(self, tag: str) -> str:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    MALWARE_BAZAAR_API,
                    data={"query": "get_taginfo", "tag": tag, "limit": 10},
                )
                resp.raise_for_status()
                result: Dict[str, Any] = resp.json()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        if result.get("query_status") != "ok":
            return json.dumps({"found": False})

        data = result.get("data", [])
        summaries = [
            {
                "sha256_hash": s.get("sha256_hash"),
                "file_name": s.get("file_name"),
                "signature": s.get("signature"),
            }
            for s in data[:10]
        ]
        return json.dumps({"found": True, "count": len(data), "samples": summaries})

    def _query_signature(self, signature: str) -> str:
        return self._query_tag(signature)


class VirusTotalTool(_BaseToolMixin, BaseTool if LANGCHAIN_AVAILABLE else object):  # type: ignore[misc]
    """Scan IOCs against VirusTotal v3 API.

    Requires VIRUSTOTAL_API_KEY environment variable.
    """

    name: str = "virus_total_scanner"
    description: str = (
        "Scan an IOC (ip, domain, url, hash) against VirusTotal. "
        "Requires VIRUSTOTAL_API_KEY env var. "
        "Input: JSON with 'ioc_type' and 'ioc_value'. "
        "Returns: detection stats and reputation data."
    )

    api_key: str = Field(default_factory=lambda: os.getenv("VIRUSTOTAL_API_KEY", ""))

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key:
            return json.dumps({"error": "VIRUSTOTAL_API_KEY env var not set"})

        data = self._safe_json(query)
        ioc_type = data.get("ioc_type", "")
        ioc_value = data.get("ioc_value", "")

        endpoints: Dict[str, str] = {
            "ip": f"ip_addresses/{ioc_value}",
            "domain": f"domains/{ioc_value}",
            "hash": f"files/{ioc_value}",
            "url": f"urls/{self._url_id(ioc_value)}",
        }
        endpoint = endpoints.get(ioc_type)
        if endpoint is None:
            return json.dumps(
                {"error": f"IOC type '{ioc_type}' not supported by VirusTotal"}
            )

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{VIRUSTOTAL_API}{endpoint}",
                    headers={"x-apikey": self.api_key, "accept": "application/json"},
                )
                resp.raise_for_status()
                result: Dict[str, Any] = resp.json()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        attrs = result.get("data", {}).get("attributes", {})
        stats = attrs.get("last_analysis_stats", {})
        return json.dumps(
            {
                "found": True,
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "undetected": stats.get("undetected", 0),
                "total_engines": sum(stats.values()),
                "reputation": attrs.get("reputation", 0),
                "tags": attrs.get("tags", []),
                "last_analysis_date": attrs.get("last_analysis_date"),
            }
        )

    @staticmethod
    def _url_id(url: str) -> str:
        """VirusTotal URL identifier: base64url of the URL (no padding)."""
        import base64

        return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


class AbuseIPDBTool(_BaseToolMixin, BaseTool if LANGCHAIN_AVAILABLE else object):  # type: ignore[misc]
    """Check IP reputation against AbuseIPDB v2 API.

    Requires ABUSEIPDB_API_KEY environment variable.
    """

    name: str = "abuse_ipdb_scanner"
    description: str = (
        "Scan an IP address against AbuseIPDB for abuse reports. "
        "Requires ABUSEIPDB_API_KEY env var. "
        "Input: JSON with 'ip' (the IP address) and optional 'max_age_days'. "
        "Returns: abuse confidence score, report count, ISP, country."
    )

    api_key: str = Field(default_factory=lambda: os.getenv("ABUSEIPDB_API_KEY", ""))

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not self.api_key:
            return json.dumps({"error": "ABUSEIPDB_API_KEY env var not set"})

        data = self._safe_json(query)
        ip = data.get("ip", data.get("value", ""))
        max_age = data.get("max_age_days", 90)

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    ABUSEIPDB_API,
                    headers={"Key": self.api_key, "Accept": "application/json"},
                    params={
                        "ipAddress": ip,
                        "maxAgeInDays": str(max_age),
                        "verbose": "",
                    },
                )
                resp.raise_for_status()
                result: Dict[str, Any] = resp.json()
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        ip_data = result.get("data", {})
        return json.dumps(
            {
                "found": True,
                "ip": ip_data.get("ipAddress"),
                "abuse_confidence_score": ip_data.get("abuseConfidenceScore", 0),
                "total_reports": ip_data.get("totalReports", 0),
                "last_reported_at": ip_data.get("lastReportedAt"),
                "country": ip_data.get("countryCode"),
                "isp": ip_data.get("isp"),
                "domain": ip_data.get("domain"),
                "usage_type": ip_data.get("usageType"),
            }
        )


class MitreAttackTool(_BaseToolMixin, BaseTool if LANGCHAIN_AVAILABLE else object):  # type: ignore[misc]
    """Map malware families and techniques to MITRE ATT&CK framework.

    No API key required. Uses built-in mappings for common families.
    """

    name: str = "mitre_attack_mapper"
    description: str = (
        "Map a malware family name or signature to MITRE ATT&CK tactics "
        "and techniques. Input: JSON with 'query_type' (malware|technique|group) "
        "and 'value'. Returns: matched techniques with tactic and name."
    )

    # Built-in mapping — extended with common families
    _malware_map: Dict[str, List[Dict[str, str]]] = {
        "emotet": [
            {"tactic": "Initial Access", "technique": "T1566", "name": "Phishing"},
            {"tactic": "Execution", "technique": "T1204", "name": "User Execution"},
            {
                "tactic": "Persistence",
                "technique": "T1547",
                "name": "Boot or Logon Autostart Execution",
            },
            {
                "tactic": "Defense Evasion",
                "technique": "T1027",
                "name": "Obfuscated Files or Information",
            },
            {
                "tactic": "Command and Control",
                "technique": "T1071",
                "name": "Application Layer Protocol",
            },
        ],
        "trickbot": [
            {"tactic": "Initial Access", "technique": "T1566", "name": "Phishing"},
            {"tactic": "Execution", "technique": "T1204", "name": "User Execution"},
            {
                "tactic": "Defense Evasion",
                "technique": "T1055",
                "name": "Process Injection",
            },
            {
                "tactic": "Credential Access",
                "technique": "T1003",
                "name": "OS Credential Dumping",
            },
            {
                "tactic": "Lateral Movement",
                "technique": "T1021",
                "name": "Remote Services",
            },
            {
                "tactic": "Command and Control",
                "technique": "T1573",
                "name": "Encrypted Channel",
            },
        ],
        "cobalt strike": [
            {
                "tactic": "Execution",
                "technique": "T1059",
                "name": "Command and Scripting Interpreter",
            },
            {
                "tactic": "Defense Evasion",
                "technique": "T1055",
                "name": "Process Injection",
            },
            {
                "tactic": "Command and Control",
                "technique": "T1071",
                "name": "Application Layer Protocol",
            },
            {
                "tactic": "Command and Control",
                "technique": "T1573",
                "name": "Encrypted Channel",
            },
            {
                "tactic": "Credential Access",
                "technique": "T1003",
                "name": "OS Credential Dumping",
            },
        ],
        "ransomware": [
            {
                "tactic": "Impact",
                "technique": "T1486",
                "name": "Data Encrypted for Impact",
            },
            {
                "tactic": "Defense Evasion",
                "technique": "T1562",
                "name": "Impair Defenses",
            },
            {
                "tactic": "Discovery",
                "technique": "T1083",
                "name": "File and Directory Discovery",
            },
            {
                "tactic": "Exfiltration",
                "technique": "T1041",
                "name": "Exfiltration Over C2 Channel",
            },
        ],
        "qakbot": [
            {"tactic": "Initial Access", "technique": "T1566", "name": "Phishing"},
            {
                "tactic": "Defense Evasion",
                "technique": "T1055",
                "name": "Process Injection",
            },
            {
                "tactic": "Credential Access",
                "technique": "T1003",
                "name": "OS Credential Dumping",
            },
            {
                "tactic": "Command and Control",
                "technique": "T1071",
                "name": "Application Layer Protocol",
            },
        ],
        "agent tesla": [
            {"tactic": "Initial Access", "technique": "T1566", "name": "Phishing"},
            {
                "tactic": "Execution",
                "technique": "T1204",
                "name": "User Execution",
            },
            {
                "tactic": "Credential Access",
                "technique": "T1003",
                "name": "OS Credential Dumping",
            },
            {
                "tactic": "Credential Access",
                "technique": "T1555",
                "name": "Credentials from Password Stores",
            },
            {
                "tactic": "Exfiltration",
                "technique": "T1041",
                "name": "Exfiltration Over C2 Channel",
            },
        ],
        "lokibot": [
            {"tactic": "Initial Access", "technique": "T1566", "name": "Phishing"},
            {
                "tactic": "Credential Access",
                "technique": "T1003",
                "name": "OS Credential Dumping",
            },
            {
                "tactic": "Exfiltration",
                "technique": "T1041",
                "name": "Exfiltration Over C2 Channel",
            },
        ],
        "redline stealer": [
            {
                "tactic": "Credential Access",
                "technique": "T1555",
                "name": "Credentials from Password Stores",
            },
            {
                "tactic": "Credential Access",
                "technique": "T1539",
                "name": "Steal Web Session Cookie",
            },
            {
                "tactic": "Collection",
                "technique": "T1005",
                "name": "Data from Local System",
            },
            {
                "tactic": "Exfiltration",
                "technique": "T1041",
                "name": "Exfiltration Over C2 Channel",
            },
        ],
        "njrat": [
            {
                "tactic": "Execution",
                "technique": "T1204",
                "name": "User Execution",
            },
            {
                "tactic": "Persistence",
                "technique": "T1547",
                "name": "Boot or Logon Autostart Execution",
            },
            {
                "tactic": "Command and Control",
                "technique": "T1071",
                "name": "Application Layer Protocol",
            },
            {
                "tactic": "Collection",
                "technique": "T1113",
                "name": "Screen Capture",
            },
        ],
    }

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        data = self._safe_json(query)
        qtype = data.get("query_type", "malware")
        value = data.get("value", "").lower().strip()

        if qtype == "malware":
            return self._lookup_malware(value)
        elif qtype == "technique":
            return self._lookup_technique(value)
        elif qtype == "group":
            return self._lookup_group(value)
        else:
            return json.dumps({"error": f"Unknown query_type: {qtype}"})

    # ------------------------------------------------------------------
    def _lookup_malware(self, name: str) -> str:
        # Exact match
        if name in self._malware_map:
            return json.dumps(
                {"found": True, "malware": name, "techniques": self._malware_map[name]}
            )
        # Fuzzy substring match
        for key, techniques in self._malware_map.items():
            if key in name or name in key:
                return json.dumps(
                    {"found": True, "malware": key, "techniques": techniques}
                )
        return json.dumps(
            {"found": False, "message": f"No MITRE mapping for '{name}'"}
        )

    def _lookup_technique(self, technique_id: str) -> str:
        tid = technique_id.upper()
        for malware, techniques in self._malware_map.items():
            for t in techniques:
                if t["technique"].upper() == tid:
                    return json.dumps(
                        {
                            "found": True,
                            "technique": t,
                            "associated_malware": malware,
                        }
                    )
        return json.dumps(
            {"found": False, "message": f"Technique {tid} not found in built-in map"}
        )

    def _lookup_group(self, group_name: str) -> str:
        return json.dumps(
            {
                "found": False,
                "message": (
                    "Group lookup requires the MITRE ATT&CK TAXII server. "
                    "Use the built-in malware lookup instead."
                ),
            }
        )


# ============================================================================
# Scanner registry — deterministic routing: IOC type → applicable scanners
# ============================================================================

SCANNER_REGISTRY: Dict[IOCType, List[type]] = {
    IOCType.IP: [VirusTotalTool, AbuseIPDBTool],
    IOCType.DOMAIN: [VirusTotalTool],
    IOCType.URL: [VirusTotalTool],
    IOCType.HASH: [MalwareBazaarTool, VirusTotalTool, MitreAttackTool],
    IOCType.EMAIL: [],
}


def _get_scanners_for_ioc(ioc_type: IOCType) -> List[Any]:
    """Instantiate applicable scanners for a given IOC type."""
    scanner_classes = SCANNER_REGISTRY.get(ioc_type, [])
    scanners: List[Any] = []
    for cls in scanner_classes:
        try:
            scanners.append(cls())
        except Exception:
            pass
    return scanners


# ============================================================================
# Multi-Agent Orchestrator
# ============================================================================


class MultiAgentScanner:
    """Multi-agent IOC scanner with two modes:

    - **Deterministic** (default): routes to all applicable scanners based on
      IOC type, runs them in parallel, collects findings.
    - **LLM Agent**: uses a LangChain AgentExecutor with all four tools;
      the LLM decides which tools to invoke and interprets results.

    Parameters
    ----------
    use_llm : bool
        Enable LLM-driven agent mode (requires langchain + langchain_openai).
    model_name : str
        LLM model name (default ``gpt-4o-mini``).
    openai_api_key : Optional[str]
        Override OPENAI_API_KEY env var.
    openai_base_url : Optional[str]
        Override OPENAI_BASE_URL (for local models e.g. Ollama / LM Studio).
    temperature : float
        LLM temperature (0.0 = deterministic).
    """

    def __init__(
        self,
        use_llm: bool = False,
        model_name: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.use_llm = use_llm
        self.model_name = model_name
        self.temperature = temperature
        self.llm: Any = None

        if use_llm:
            if not CHAT_AVAILABLE:
                warnings.warn(
                    "langchain_openai not installed — falling back to deterministic mode."
                )
                self.use_llm = False
            else:
                api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "not-needed")
                base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
                kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "temperature": temperature,
                    "api_key": api_key,
                }
                if base_url:
                    kwargs["base_url"] = base_url
                self.llm = ChatOpenAI(**kwargs)

    # ------------------------------------------------------------------
    def scan(self, ioc_type: str, ioc_value: str) -> ScanReport:
        """Run all applicable scanners for an IOC and return a
        :class:`ScanReport` with aggregated findings.

        Parameters
        ----------
        ioc_type : str
            One of ``ip``, ``domain``, ``url``, ``hash``, ``email``.
        ioc_value : str
            The IOC value.

        Returns
        -------
        ScanReport
            Contains ``.findings`` (list of ScanFinding), ``.summary`` (str),
            ``.to_list()`` (list of dicts), and ``.raw_results`` (dict).
        """
        ioc_type_enum = IOCType(ioc_type.lower())
        # Validate
        IOCInput(ioc_type=ioc_type_enum, ioc_value=ioc_value)

        report = ScanReport(ioc_type=ioc_type_enum, ioc_value=ioc_value)
        scanners = _get_scanners_for_ioc(ioc_type_enum)

        if not scanners:
            report.findings.append(
                ScanFinding(
                    engine=ScanEngine.MITRE_ATTACK,
                    key="info",
                    value=f"No scanners available for IOC type '{ioc_type}'",
                    severity="low",
                )
            )
            return report

        # ---- Run each scanner -------------------------------------------------
        for scanner in scanners:
            engine_name = ScanEngine(scanner.name.replace("_scanner", ""))
            try:
                # Build a unified query dict
                query = {
                    "query_type": self._infer_query_type(scanner, ioc_type_enum),
                    "value": ioc_value,
                    "ioc_type": ioc_type,
                    "ioc_value": ioc_value,
                    "ip": ioc_value,
                }
                raw = scanner._run(json.dumps(query))
                report.raw_results[engine_name.value] = json.loads(raw)
                self._parse_findings(report, engine_name, raw)
            except Exception as exc:
                report.findings.append(
                    ScanFinding(
                        engine=engine_name,
                        key="error",
                        value=str(exc),
                        severity="low",
                    )
                )

        # ---- Optional LLM analysis -------------------------------------------
        if self.use_llm and self.llm and report.findings:
            try:
                analysis = self._llm_analyze(report)
                report.findings.append(
                    ScanFinding(
                        engine=ScanEngine.MITRE_ATTACK,
                        key="llm_analysis",
                        value=analysis,
                        severity="medium",
                    )
                )
            except Exception:
                pass

        return report

    # ------------------------------------------------------------------
    def scan_with_agent(self, ioc_type: str, ioc_value: str) -> str:
        """Full LangChain agent-driven scan.

        The LLM decides which tools to query and how to interpret results.
        Requires ``use_llm=True`` at construction.
        """
        if not AGENT_AVAILABLE or not self.llm:
            return (
                "Agent mode requires langchain, langchain_openai, and an LLM. "
                "Use scan() for deterministic mode."
            )

        tools = [
            MalwareBazaarTool(),
            VirusTotalTool(),
            AbuseIPDBTool(),
            MitreAttackTool(),
        ]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a cybersecurity threat intelligence analyst. "
                        "Given an IOC (Indicator of Compromise), determine which "
                        "tools to query and interpret the results.\n\n"
                        "Available tools:\n"
                        "- malware_bazaar_scanner: query file hashes, tags, signatures.\n"
                        "- virus_total_scanner: scan IPs, domains, URLs, hashes.\n"
                        "- abuse_ipdb_scanner: check IP reputation.\n"
                        "- mitre_attack_mapper: map malware names/signatures to MITRE ATT&CK.\n\n"
                        "For hashes: query malware_bazaar AND virus_total. "
                        "For IPs: query virus_total AND abuse_ipdb. "
                        "For domains/URLs: query virus_total. "
                        "If malware_bazaar returns a signature, query mitre_attack_mapper "
                        "with that signature name.\n"
                        "Always provide a concise summary of all findings."
                    ),
                ),
                (
                    "human",
                    "Scan this IOC — type: {ioc_type}, value: {ioc_value}",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(self.llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=10,
            handle_parsing_errors=True,
        )

        result = executor.invoke({"ioc_type": ioc_type, "ioc_value": ioc_value})
        return result.get("output", str(result))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_query_type(scanner: Any, ioc_type: IOCType) -> str:
        """Return the ``query_type`` string a specific scanner expects."""
        if isinstance(scanner, MalwareBazaarTool):
            return "hash"
        if isinstance(scanner, VirusTotalTool):
            return ioc_type.value
        if isinstance(scanner, AbuseIPDBTool):
            return "ip"
        if isinstance(scanner, MitreAttackTool):
            return "malware"
        return ioc_type.value

    @staticmethod
    def _parse_findings(
        report: ScanReport, engine: ScanEngine, raw_result: str
    ) -> None:
        """Parse a raw JSON string into :class:`ScanFinding` objects."""
        try:
            data: Dict[str, Any] = json.loads(raw_result)
        except json.JSONDecodeError:
            return

        if "error" in data:
            report.findings.append(
                ScanFinding(
                    engine=engine,
                    key="error",
                    value=data["error"],
                    severity="low",
                )
            )
            return

        # --- MalwareBazaar ----------------------------------------------------
        if engine == ScanEngine.MALWARE_BAZAAR:
            if data.get("found"):
                sig = data.get("signature") or "None"
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="signature",
                        value=sig,
                        severity="high" if sig != "None" else "medium",
                    )
                )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="file_type",
                        value=data.get("file_type", "Unknown"),
                    )
                )
                tags: List[str] = data.get("tags", [])
                if tags:
                    report.findings.append(
                        ScanFinding(
                            engine=engine,
                            key="tags",
                            value=", ".join(tags),
                            severity="medium",
                        )
                    )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="first_seen",
                        value=data.get("first_seen", "Unknown"),
                    )
                )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="last_seen",
                        value=data.get("last_seen", "Unknown"),
                    )
                )
            else:
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="result",
                        value="Not found in MalwareBazaar",
                        severity="low",
                    )
                )

        # --- VirusTotal -------------------------------------------------------
        elif engine == ScanEngine.VIRUSTOTAL:
            if data.get("found"):
                malicious = data.get("malicious", 0)
                total = data.get("total_engines", 0)
                severity = (
                    "critical"
                    if malicious > 10
                    else "high" if malicious > 3 else "medium" if malicious > 0 else "low"
                )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="detection_rate",
                        value=f"{malicious}/{total} engines flagged malicious",
                        severity=severity,
                    )
                )
                if data.get("reputation") is not None:
                    report.findings.append(
                        ScanFinding(
                            engine=engine,
                            key="reputation",
                            value=str(data["reputation"]),
                        )
                    )
                vt_tags: List[str] = data.get("tags", [])
                if vt_tags:
                    report.findings.append(
                        ScanFinding(
                            engine=engine,
                            key="tags",
                            value=", ".join(vt_tags),
                        )
                    )
            else:
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="result",
                        value="Not found in VirusTotal",
                        severity="low",
                    )
                )

        # --- AbuseIPDB --------------------------------------------------------
        elif engine == ScanEngine.ABUSEIPDB:
            if data.get("found"):
                score = data.get("abuse_confidence_score", 0)
                severity = (
                    "critical"
                    if score > 80
                    else "high" if score > 50 else "medium" if score > 20 else "low"
                )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="abuse_confidence",
                        value=f"{score}%",
                        severity=severity,
                    )
                )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="total_reports",
                        value=str(data.get("total_reports", 0)),
                    )
                )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="country",
                        value=data.get("country", "Unknown"),
                    )
                )
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="isp",
                        value=data.get("isp", "Unknown"),
                    )
                )

        # --- MITRE ATT&CK -----------------------------------------------------
        elif engine == ScanEngine.MITRE_ATTACK:
            techniques = data.get("techniques", [])
            if techniques:
                for t in techniques:
                    report.findings.append(
                        ScanFinding(
                            engine=engine,
                            key="mitre_technique",
                            value=f"{t['technique']} — {t['name']}  [{t['tactic']}]",
                            severity="medium",
                        )
                    )
            elif not data.get("found"):
                report.findings.append(
                    ScanFinding(
                        engine=engine,
                        key="result",
                        value=data.get("message", "No MITRE mapping found"),
                        severity="low",
                    )
                )

    def _llm_analyze(self, report: ScanReport) -> str:
        """Ask the LLM to produce a threat assessment from the findings."""
        if not self.llm:
            return "LLM not available"

        from langchain_core.messages import HumanMessage, SystemMessage

        findings_text = "\n".join(
            f"[{f.engine.value}] {f.key}: {f.value}  (severity: {f.severity or 'n/a'})"
            for f in report.findings
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a senior threat intelligence analyst. "
                    "Review the scan findings below and provide a brief "
                    "threat assessment with recommended actions."
                )
            ),
            HumanMessage(
                content=(
                    f"IOC: {report.ioc_type.value} — {report.ioc_value}\n\n"
                    f"Scan findings:\n{findings_text}\n\n"
                    "Provide a concise threat assessment."
                )
            ),
        ]

        response = self.llm.invoke(messages)
        return getattr(response, "content", str(response))


# ============================================================================
# Convenience function
# ============================================================================


def scan_ioc(ioc_type: str, ioc_value: str, use_llm: bool = False) -> ScanReport:
    """Scan an IOC across multiple threat-intelligence engines.

    One-call entry point. Returns a :class:`ScanReport` whose ``.to_list()``
    method gives the "list of information" for downstream consumption.

    Parameters
    ----------
    ioc_type : str
        ``ip`` | ``domain`` | ``url`` | ``hash`` | ``email``
    ioc_value : str
        The IOC to scan.
    use_llm : bool
        Whether to append an LLM analysis to the findings.

    Returns
    -------
    ScanReport

    Example
    -------
    >>> report = scan_ioc("hash", "2b8ba28ca8ce0c5d3b6f25b0e4d2e8a6")
    >>> print(report.summary)
    >>> for info in report.to_list():
    ...     print(info)
    """
    scanner = MultiAgentScanner(use_llm=use_llm)
    return scanner.scan(ioc_type, ioc_value)


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python multi-engine_scanner.py <ioc_type> <ioc_value>\n"
            "  ioc_type: ip | domain | url | hash | email\n"
            "  ioc_value: the IOC to scan\n"
            "\nExample:\n"
            "  python multi-engine_scanner.py ip 8.8.8.8\n"
            "  python multi-engine_scanner.py hash 2b8ba28ca8ce0c5d3b6f25b0e4d2e8a6\n"
            "\nAPI keys (env vars):\n"
            "  VIRUSTOTAL_API_KEY   — VirusTotal v3\n"
            "  ABUSEIPDB_API_KEY    — AbuseIPDB v2\n"
            "  OPENAI_API_KEY       — LLM agent mode (optional)\n"
            "  MalwareBazaar & MITRE ATT&CK — no key needed."
        )
        sys.exit(1)

    _ioc_type, _ioc_value = sys.argv[1], sys.argv[2]
    _report = scan_ioc(_ioc_type, _ioc_value)
    print(_report.summary)
    print("\n" + "─" * 60)
    print("Raw results:")
    for _engine, _data in _report.raw_results.items():
        print(f"\n  [{_engine}]")
        print(f"  {json.dumps(_data, indent=2)}")
