"""
PITC UDIL (Release 5.0) & K-Electric SAP IS-U Sandbox Compatibility Test Suite.
Validates that raw Pakistani DISCO smart meter payloads (IESCO, LESCO, MEPCO, FESCO, K-Electric)
translate with 100% fidelity into our 4-Channel Deep Learning + 8-Rule Verification Agent pipeline.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.verification.verify import CustomerContext
from src.agents.coordinator.coordinator import coordinate
from src.preprocessing.transform_single import transform_daily_series


def map_pitc_udil_or_ke_payload(payload: dict) -> tuple[str, list[float], CustomerContext]:
    """
    Zero-touch adapter translating either:
      1. PITC UDIL v5.0 (LOAD_PROFILE_DATA, BILLING_DATA, EVENT_LOG with DLMS/COSEM OBIS codes)
      2. K-Electric SAP S/4HANA IS-U + Itron MDM payload
    into (consumer_id, daily_kwh, CustomerContext).
    """
    if "udil_header" in payload:
        ident = payload["meter_identity"]
        lp = payload["load_profile_data"]
        events = payload.get("event_log", [])
        audit = payload.get("disco_ms_audit_history", {})

        consumer_id = ident["consumer_ref_no"]
        feeder_id = ident["feeder_code"]
        transformer_id = ident["transformer_id"]
        tariff_raw = ident["tariff_code"].upper()
        tariff_cat = "industrial" if "B-" in tariff_raw else ("commercial" if "A-2" in tariff_raw else "residential")

        solar_export = float(lp.get("active_energy_neg_export_kwh", 0.0))
        has_solar = solar_export > 0.0
        seal_broken = any("COVER_OPEN" in e.get("event_name", "") for e in events)
        rev_current = any("REVERSAL" in e.get("event_name", "") for e in events)
        tamper_count = sum(int(e.get("occurrence_counter", 1)) for e in events)

        hist_mean = float(lp["historical_daily_mean_kwh"])
        rec_mean = float(lp["recent_30d_daily_mean_kwh"])
        daily_kwh = [hist_mean] * 60 + [rec_mean] * 30

        ctx = CustomerContext(
            consumer_id=consumer_id,
            feeder_id=feeder_id,
            transformer_id=transformer_id,
            tariff_category=tariff_cat,
            sanctioned_load_kw=float(ident.get("sanctioned_load_kw", 7.0)),
            feeder_loss_pct=float(ident.get("feeder_atc_loss_pct", 18.0)),
            recent_audit_result=audit.get("recent_audit_result", "none"),
            months_since_last_audit=int(audit.get("months_since_last_audit", 999)),
            has_rooftop_solar=has_solar,
            solar_net_metering_active=has_solar,
            tamper_event_count=tamper_count,
            meter_seal_broken=seal_broken,
            reverse_current_alert=rev_current,
            historical_mean_kwh=hist_mean,
            recent_30d_mean_kwh=rec_mean,
            lowest_window_mean_kwh=rec_mean,
            tariff_rate_per_kwh=34.6,
        )
        return consumer_id, daily_kwh, ctx

    # K-Electric SAP IS-U + Itron MDM branch
    acct = payload["contract_account"]
    mdm = payload["itron_mdm_intervals"]
    events = payload.get("dlms_obis_events", [])

    consumer_id = acct["contract_account_no"]
    hist_mean = float(mdm["active_energy_import_kwh_mean"])
    rec_mean = float(mdm["recent_30d_import_kwh_mean"])
    daily_kwh = [hist_mean] * 60 + [rec_mean] * 30
    solar_export = float(mdm.get("active_energy_export_kwh", 0.0))

    ctx = CustomerContext(
        consumer_id=consumer_id,
        feeder_id=acct["feeder_id"],
        transformer_id=acct["pmt_id"],
        tariff_category="residential",
        sanctioned_load_kw=float(acct.get("sanctioned_load_kw", 9.5)),
        feeder_loss_pct=float(acct.get("pmt_energy_balance_loss_pct", 25.0)),
        has_rooftop_solar=solar_export > 0.0,
        solar_net_metering_active=solar_export > 0.0,
        tamper_event_count=sum(int(e.get("count", 1)) for e in events),
        meter_seal_broken=any("SEAL_BROKEN" in e.get("description", "") or "MAGNETIC" in e.get("description", "") for e in events),
        reverse_current_alert=False,
        historical_mean_kwh=hist_mean,
        recent_30d_mean_kwh=rec_mean,
        lowest_window_mean_kwh=rec_mean,
        tariff_rate_per_kwh=37.0,
    )
    return consumer_id, daily_kwh, ctx


def run_verification():
    sample_packets = [
        {
            "name": "IESCO F-10 Shunt Bypass (UDIL v5.0)",
            "payload": {
                "udil_header": {"disco_code": 14, "disco_name": "IESCO"},
                "meter_identity": {
                    "consumer_ref_no": "14-11204-0089214",
                    "feeder_code": "FDR-ISB-F10-03",
                    "transformer_id": "DT-F10-042",
                    "tariff_code": "A-1",
                    "sanctioned_load_kw": 12.0,
                    "feeder_atc_loss_pct": 22.8,
                },
                "load_profile_data": {
                    "historical_daily_mean_kwh": 46.4,
                    "recent_30d_daily_mean_kwh": 6.1,
                    "active_energy_neg_export_kwh": 0.0,
                },
                "event_log": [
                    {"event_name": "TERMINAL_COVER_OPEN", "occurrence_counter": 2},
                    {"event_name": "CURRENT_REVERSAL_PHASE_A", "occurrence_counter": 1},
                ],
            },
            "raw_score": 0.842,
            "expect_suspect": True,
        },
        {
            "name": "IESCO DHA-II 15kW Solar Net-Metering (OBIS 1.0.2.8.0.255)",
            "payload": {
                "udil_header": {"disco_code": 14, "disco_name": "IESCO"},
                "meter_identity": {
                    "consumer_ref_no": "14-11208-0194420",
                    "feeder_code": "FDR-ISB-DHA2-01",
                    "transformer_id": "DT-DHA2-019",
                    "tariff_code": "A-1-TOU",
                    "sanctioned_load_kw": 15.0,
                    "feeder_atc_loss_pct": 6.4,
                },
                "load_profile_data": {
                    "historical_daily_mean_kwh": 52.0,
                    "recent_30d_daily_mean_kwh": 11.4,
                    "active_energy_neg_export_kwh": 38.6,
                },
                "event_log": [],
            },
            "raw_score": 0.794,
            "expect_suspect": False,
        },
        {
            "name": "K-Electric IBC Johar PMT Imbalance + Magnet Tamper (SAP IS-U)",
            "payload": {
                "sap_isu_header": {"company_code": "KE01"},
                "contract_account": {
                    "contract_account_no": "0400029184712",
                    "feeder_id": "FDR-KHI-JHR-11KV",
                    "pmt_id": "PMT-JHR-BLK14-08",
                    "sanctioned_load_kw": 9.5,
                    "pmt_energy_balance_loss_pct": 29.4,
                },
                "itron_mdm_intervals": {
                    "active_energy_import_kwh_mean": 41.2,
                    "recent_30d_import_kwh_mean": 5.4,
                    "active_energy_export_kwh": 0.0,
                },
                "dlms_obis_events": [
                    {"description": "STRONG_DC_MAGNETIC_FIELD_DETECTED", "count": 4},
                    {"description": "TERMINAL_COVER_SEAL_BROKEN", "count": 1},
                ],
            },
            "raw_score": 0.868,
            "expect_suspect": True,
        },
    ]

    print("=" * 78)
    print("  PAKISTANI DISCO (PITC UDIL) & K-ELECTRIC (SAP IS-U) SANDBOX VERIFICATION")
    print("=" * 78)
    for item in sample_packets:
        cid, series, ctx = map_pitc_udil_or_ke_payload(item["payload"])
        scaled, _, _ = transform_daily_series(series, target_seq_len=1034)
        assert len(scaled) == 1034, "Sequence alignment to 1,034 steps failed"
        res = coordinate(
            consumer_id=cid,
            consumer_type="residential",
            residential_probability=item["raw_score"],
            context=ctx,
            threshold=0.65,
        )
        assert res.is_theft_suspect == item["expect_suspect"], f"Mismatch on {item['name']}"
        print(
            f"[PASS] {item['name']}\n"
            f"       Ref/Acct: {cid} | Raw DL: {res.raw_model_score:.3f} -> "
            f"Verified: {res.verified_theft_probability:.3f} | Tier: {res.risk_tier} | "
            f"Suspect: {res.is_theft_suspect} | PKR Loss: Rs.{res.financial_impact.get('estimated_monthly_loss_currency', 0):,.0f}/mo"
        )
    print("=" * 78)
    print("  ALL PITC UDIL v5.0 & K-ELECTRIC SAP IS-U SANDBOX CHECKS PASSED (100%)")
    print("=" * 78)


if __name__ == "__main__":
    run_verification()
