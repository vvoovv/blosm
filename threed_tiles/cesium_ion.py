"""Cesium ion endpoint resolution, independent of Blender and tile importing."""
from dataclasses import dataclass, field
import json
from urllib.parse import urlsplit

from ..downloads import Request
from ..downloads.diagnostics import redact_text


GOOGLE_ASSET_ID = 2275207
API_URL = 'https://api.cesium.com/v1/assets/{asset_id}/endpoint'


class AssetResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Attribution:
    html: str
    collapsible: bool = True


@dataclass(frozen=True)
class AssetEndpoint:
    """Per-operation access information; never write provider credentials to prefs."""
    asset_id: int
    url: str = field(repr=False)
    headers: dict = field(default_factory=dict, repr=False)
    attributions: tuple = field(default_factory=tuple, repr=False)
    external_type: str = ''


def endpoint_request(token, asset_id=GOOGLE_ASSET_ID):
    if not isinstance(asset_id, int) or isinstance(asset_id, bool) or asset_id <= 0:
        raise ValueError('A positive Cesium ion asset ID is required')
    token = token.strip()
    if not token:
        raise AssetResolutionError('A Cesium ion access token is required')
    if any(char.isspace() for char in token):
        raise AssetResolutionError('The Cesium ion access token must not contain whitespace')
    return Request(API_URL.format(asset_id=asset_id), headers={
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/json',
        'User-Agent': 'Blosm',
    }, decode_gzip=True, timeout=30.)


def parse_endpoint(content, asset_id=GOOGLE_ASSET_ID):
    if isinstance(content, (str, bytes, bytearray)) and len(content) > 1024 * 1024:
        raise AssetResolutionError('Cesium ion endpoint response exceeds the 1 MiB size limit')
    try:
        data = json.loads(content)
    except (ValueError, UnicodeError, RecursionError, TypeError):
        raise AssetResolutionError('Cesium ion returned invalid JSON') from None
    if not isinstance(data, dict) or data.get('type') != '3DTILES':
        raise AssetResolutionError('Cesium ion did not return a 3D Tiles endpoint')

    external_type = data.get('externalType', '')
    headers = {}
    if external_type:
        if external_type != '3DTILES' or not isinstance(data.get('options'), dict):
            raise AssetResolutionError('Cesium ion returned an unsupported external provider')
        url = data['options'].get('url')
    else:
        url = data.get('url')
        access_token = data.get('accessToken')
        if not isinstance(access_token, str) or not access_token or any(c.isspace() for c in access_token):
            raise AssetResolutionError('Cesium ion did not return valid asset credentials')
        headers['Authorization'] = 'Bearer ' + access_token

    try:
        parts = urlsplit(url) if isinstance(url, str) else None
        valid_url = (parts is not None and parts.scheme == 'https' and parts.hostname
                     and not parts.username and not parts.password and not parts.fragment
                     and not any(c.isspace() for c in url))
        if parts is not None:
            parts.port  # Validate a malformed port without exposing the returned URL.
    except ValueError:
        valid_url = False
    if not valid_url:
        raise AssetResolutionError('Cesium ion did not return a valid HTTPS root tileset URL')

    credits = data.get('attributions', [])
    if not isinstance(credits, list) or any(
            not isinstance(credit, dict) or not isinstance(credit.get('html'), str)
            or not isinstance(credit.get('collapsible', True), bool) for credit in credits):
        raise AssetResolutionError('Cesium ion returned invalid attribution information')
    attributions = tuple(Attribution(c['html'], c.get('collapsible', True)) for c in credits)
    return AssetEndpoint(asset_id, url, headers, attributions, external_type)


def resolve_asset(token, asset_id=GOOGLE_ASSET_ID):
    """Yield one endpoint request; return a descriptor without fetching any tiles."""
    request = endpoint_request(token, asset_id)
    try:
        content = yield request
    except Exception as exc:
        message = redact_text(exc, request.url, request.headers)
        raise AssetResolutionError('Unable to resolve Cesium ion access: ' + message) from None
    return parse_endpoint(content, asset_id)
