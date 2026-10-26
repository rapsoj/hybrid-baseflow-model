import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import os

logger = logging.getLogger(__name__)


class _TimeoutAdapter(HTTPAdapter):
    """Default timeout for every request (some HDE calls were made without one and could hang)."""

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 60
        return super().send(request, **kwargs)


def make_session() -> requests.Session:
    """
    Session that retries with exponential backoff on timeouts, 5xx, 429 and 403. The Hydrology API answers 403/timeouts
    when it is hit too hard, so those are treated as "try again later" instead of failing the borehole.
    """
    retry = Retry(
        total=8, connect=8, read=8, status=8, backoff_factor=2,
        status_forcelist=(403, 429, 500, 502, 503, 504), allowed_methods=("GET",), respect_retry_after_header=True,
    )
    sess = requests.Session()
    adapter = _TimeoutAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def station_has_15min_flow(station_guid, session=None, timeout=10):
    base = "https://environment.data.gov.uk/hydrology/id"
    url = f"{base}/stations/{station_guid}/measures?observedProperty=waterFlow"
    get = session.get if session is not None else requests.get

    try:
        r = get(url, timeout=timeout)
        if r.status_code == 404:
            return False
        r.raise_for_status()
    except Exception:
        return False

    measures = r.json().get("items", []) or []
    return any(
        str(m.get("period")) == "900" or "-t-900-" in (m.get("@id", "") or "")
        for m in measures
    )


def _download_hde_measure_data(measure_id, label, output_prefix, file_prefix, observed_property, period_label, session=None, update_existing=False):

    # build target folder + filename up-front so we can skip if file exists
    folder = os.path.join("data", "hde", output_prefix)
    fname = os.path.join(
        folder,
        f"{file_prefix}_{observed_property}_{period_label}.csv"
    )
    
    existing_df = None
    last_timestamp = None

    if os.path.exists(fname) and not update_existing:
        logger.info(f"Data already downloaded, skipping: {fname}")
        return
    
    if os.path.exists(fname):
        try:
            existing_df = pd.read_csv(fname, parse_dates=["dateTime"], index_col="dateTime")
            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                logger.debug(f"Updating existing file from {last_timestamp}: {fname}")
        except Exception:
            logger.warning(f"Failed to read existing file, will overwrite: {fname}")
            
    all_readings = []
    offset = 0
    limit = 10000
    get = session.get if session is not None else requests.get
    
    while True:
        if last_timestamp is not None:
            since = last_timestamp.isoformat()
            url = f"{measure_id}/readings?since={since}&_limit={limit}&_offset={offset}"
            logger.debug(f"Fetching new data since {last_timestamp} (offset={offset})")
        else:
            url = f"{measure_id}/readings?_limit={limit}&_offset={offset}"
            if offset % (limit * 5) == 0:
                logger.debug(f"Fetching historical data (offset={offset})")
    
        resp = get(url, timeout=30)
    
        if resp.status_code == 404:
            logger.warning(f"No readings found for {label}")
            break
    
        resp.raise_for_status()
        items = resp.json().get("items", [])
    
        if not items:
            break
    
        all_readings.extend(items)
    
        logger.debug(f"Fetched {len(items)} rows (total so far: {len(all_readings)})")
    
        if len(items) < limit:
            break
    
        offset += limit

    if not all_readings:
        logger.debug(f"No new data returned for {label}")
        return

    df = pd.DataFrame(all_readings)
    if "dateTime" in df.columns:
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df = df.set_index("dateTime").sort_index()

    df = df[[c for c in df.columns if c in ["value", "valueQualifier"]]]
    
    if last_timestamp is not None:
        df = df[df.index > last_timestamp]
    
        if df.empty:
            logger.debug(f"No new rows to append for {fname}")
            return
    
    os.makedirs(folder, exist_ok=True)
    
    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    
    tmp = fname + ".tmp"  # written then renamed, so an interrupted run never leaves a truncated csv
    df.to_csv(tmp)
    os.replace(tmp, fname)
    logger.debug(f"File now has {len(df)} rows: {fname}")
    
    
def download_hde_river_data_15min(station_guid, observed_property="waterFlow", output_prefix="", session=None, update_existing=False):
    base = "https://environment.data.gov.uk/hydrology/id"
    measures_url = f"{base}/stations/{station_guid}/measures?observedProperty={observed_property}"
    get = session.get if session is not None else requests.get
    r = get(measures_url)

    if r.status_code == 404:
        logger.warning(f"Warning: Station {station_guid} not found for {observed_property}")
        return

    r.raise_for_status()
    measures = r.json().get("items", [])

    if not measures:
        logger.warning(f"Warning: No {observed_property} measures found for station {station_guid}")
        return

    quarter_hour_measures = [m for m in measures if str(m.get("period")) == "900" or "-t-900-" in m.get("@id", "")]
    if not quarter_hour_measures:
        logger.warning(f"Warning: No 15 min {observed_property} measures for station {station_guid}")
        return

    logger.debug(f"Found {len(quarter_hour_measures)} {observed_property} measure(s) at 15 min")

    for m in quarter_hour_measures:
        measure_id = m["@id"].replace("http://", "https://")
        label = m.get("label", measure_id.split("/")[-1])

        station_name = (
            m.get("station", {})
             .get("label", "station")
             .lower()
             .replace(" ", "_")
        )

        _download_hde_measure_data(
            measure_id,
            label,
            output_prefix,        # folder stays chalk-based
            station_name,         # filename prefix
            observed_property,
            "15min",
            session=session,
            update_existing=update_existing
        )


def download_hde_rainfall_15min(station_guid, output_prefix="", session=None, update_existing=False):
    base = "https://environment.data.gov.uk/hydrology/id"
    observed_property = "rainfall"
    measures_url = f"{base}/stations/{station_guid}/measures?observedProperty=rainfall"
    get = session.get if session is not None else requests.get
    r = get(measures_url)

    if r.status_code == 404:
        logger.warning(f"Warning: Station {station_guid} not found for rainfall")
        return

    r.raise_for_status()
    measures = r.json().get("items", [])

    if not measures:
        logger.warning(f"Warning: No rainfall measures found for station {station_guid}")
        return

    quarter_hour_measures = [m for m in measures if str(m.get("period")) == "900" or "-t-900-" in m.get("@id", "")]
    if not quarter_hour_measures:
        logger.warning(f"Warning: No 15 min rainfall measures for station {station_guid}")
        return

    logger.debug(f"Found {len(quarter_hour_measures)} rainfall measure(s) at 15 min")

    for m in quarter_hour_measures:
        measure_id = m["@id"].replace("http://", "https://")
        label = m.get("label", measure_id.split("/")[-1])

        station_name = (
            m.get("station", {})
             .get("label", "station")
             .lower()
             .replace(" ", "_")
        )

        _download_hde_measure_data(
            measure_id,
            label,
            output_prefix,        # folder stays chalk-based
            station_name,         # filename prefix
            observed_property,
            "15min",
            session=session,
            update_existing=update_existing
        )


def download_hde_groundwater_15min(station_guid, output_prefix="", session=None, update_existing=False):
    base = "https://environment.data.gov.uk/hydrology/id"
    observed_property = "groundwaterLevel"
    measures_url = f"{base}/stations/{station_guid}/measures?observedProperty={observed_property}"
    get = session.get if session is not None else requests.get
    r = get(measures_url)

    if r.status_code == 404:
        logger.warning(f"Warning: Station {station_guid} not found for groundwater")
        return

    r.raise_for_status()
    measures = r.json().get("items", [])

    if not measures:
        logger.warning(f"Warning: No groundwater measures found for station {station_guid}")
        return

    # Try 15 min first
    measures_15 = [m for m in measures if str(m.get("period")) == "900" or "-t-900-" in m.get("@id", "")]
    
    if measures_15:
        selected_measures = measures_15
        period_label = "15min"
        logger.debug(f"Using 15 min groundwater data ({len(measures_15)} measure(s))")
    
    else:
        # fallback to hourly
        measures_60 = [m for m in measures if str(m.get("period")) == "3600"]
    
        if measures_60:
            selected_measures = measures_60
            period_label = "60min"
            logger.debug(f"No 15 min data — using hourly groundwater ({len(measures_60)} measure(s))")
        else:
            logger.warning(f"No suitable groundwater measures (15min or hourly) for station {station_guid}")
            return

    for m in selected_measures:
        measure_id = m["@id"].replace("http://", "https://")
        label = m.get("label", measure_id.split("/")[-1])

        station_name = (
            m.get("station", {})
             .get("label", "station")
             .lower()
             .replace(" ", "_")
        )

        _download_hde_measure_data(
            measure_id,
            label,
            output_prefix,        # folder stays chalk-based
            station_name,         # filename prefix
            observed_property,
            period_label,
            session=session,
            update_existing=update_existing
        )
