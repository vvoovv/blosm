"""Google 3D Tiles access through Cesium ion, using the shared import pipeline."""
from hashlib import sha256
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from .cesium_ion import AssetResolutionError, endpoint_request, resolve_asset
from .manager import BaseManager
from ..downloads import Request
from ..downloads.diagnostics import redact_text


def origin(url):
    parts = urlsplit(url)
    return parts.scheme.lower(), parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80)


class CreditText(HTMLParser):
    """Retain credit text and logo labels in Blender's existing attribution field."""
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.feed(html)
        self.close()

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            attrs = dict(attrs)
            self.parts.append(attrs.get('alt') or attrs.get('title') or '')

    @property
    def text(self):
        return ' '.join(' '.join(self.parts).split())


class CesiumIonManager(BaseManager):
    # Use the helper even for blocking, single-download imports, retaining
    # Request headers, verified TLS and credential-safe error handling.
    useDownloadTransport = True

    def __init__(self, rootUri, renderer):
        super().__init__('', renderer)
        self._token = ''
        self._headers = {}
        self._rootQuery = {}
        self._sessionQuery = {}
        self._origin = None
        self.resolvingIon = True

    def setToken(self, token):
        endpoint_request(token)  # Local validation before starting any work.
        self._token = token.strip()

    def iterRender(self, *bounds):
        try:
            try:
                endpoint = yield from resolve_asset(self._token)
            except AssetResolutionError as exc:
                return (str(exc),)
            self._token = ''
            self.rootUri = endpoint.url
            parts = urlsplit(endpoint.url)
            self.uriServer = urlunsplit((parts.scheme, parts.netloc, '', '', ''))
            self.baseUri = urljoin(endpoint.url, '.')
            self._origin = origin(endpoint.url)
            self._headers = endpoint.headers.copy()
            self._rootQuery = dict(parse_qsl(parts.query, keep_blank_values=True))
            self.resolvingIon = False
            for credit in endpoint.attributions:
                text = self.redactError(CreditText(credit.html).text, endpoint.url)
                self.renderer.providerCredits.append(text)
            # Resolve before prepare(): endpoint failures/cancellation must not
            # create collections, objects or scene origin properties.
            return (yield from super().iterRender(*bounds))
        finally:
            # Runtime credentials never enter scene data or addon preferences.
            self._token = ''
            self._headers.clear()
            self._rootQuery.clear()
            self._sessionQuery.clear()
            self.rootUri = self.constantUriQuery = self.uriQuery = ''
            self.resolvingIon = False

    def getUri(self, uriComponents, baseUri):
        uri = urljoin(baseUri, uriComponents.geturl())
        parts = urlsplit(uri)
        if origin(uri) == self._origin:
            # Google supplies a session in the root response. Retain it for
            # nested manifests and meshes that omit their own query string.
            self._sessionQuery.update(parse_qsl(parts.query, keep_blank_values=True))
            query = {**self._sessionQuery, **self._rootQuery}
            if 'session' in self._sessionQuery:
                query['session'] = self._sessionQuery['session']
            uri = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))
        # Absolute links to other origins retain only their own parameters.
        return uri, urljoin(uri, '.')

    def downloadRequest(self, uri):
        same_origin = self._origin is not None and origin(uri) == self._origin
        headers = self._headers.copy() if same_origin else {}
        scope, reasons = '', ()
        if same_origin:
            scope = sha256((str(sorted(self._rootQuery.items())) + str(sorted(headers.items()))
                            + urlsplit(uri).query).encode()).hexdigest()
            reasons = ('API_KEY_INVALID', 'API_KEY_EXPIRED', 'SESSION_EXPIRED', 'UNAUTHENTICATED', 'expired')
        return Request(uri, headers=headers, decode_gzip=True,
                       auth_scope=scope, auth_error_reasons=reasons)

    def redactError(self, error, uri):
        text = redact_text(error, uri, self._headers)
        return redact_text(text, self.rootUri, self._headers)
