"""
Compatibility Map

Known-good package versions per Python version.
Used as a pre-LLM step: if a package is in this table, use the known
version directly instead of asking Gemma2 to guess.

An empty string means "use latest" — safe for modern Python versions
where the package has no upper bound issues.

Merged from SmartResolver's version_resolver.py and confidence_cascade.py.
"""

from typing import Dict, Optional


# {package_name: {python_version: version_string}}
# Empty string = use latest (no pinning needed for that Python version)
COMPAT_MAP: Dict[str, Dict[str, str]] = {
    'django': {
        '2.7': '1.11.29', '3.5': '2.2.28', '3.6': '3.2.25',
        '3.7': '3.2.25',  '3.8': '',        '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'flask': {
        '2.7': '1.1.4',  '3.5': '1.1.4',  '3.6': '2.0.3',
        '3.7': '2.2.5',  '3.8': '',        '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'requests': {
        '2.7': '2.27.1', '3.5': '2.25.1', '3.6': '2.27.1',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'numpy': {
        '2.7': '1.16.6', '3.5': '1.18.5', '3.6': '1.19.5',
        '3.7': '1.21.6', '3.8': '1.24.4', '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'pandas': {
        '2.7': '0.24.2', '3.5': '0.25.3', '3.6': '1.1.5',
        '3.7': '1.3.5',  '3.8': '',        '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'scipy': {
        '2.7': '1.2.3',  '3.5': '1.4.1',  '3.6': '1.5.4',
        '3.7': '1.7.3',  '3.8': '',        '3.9': '',
    },
    'matplotlib': {
        '2.7': '2.2.5',  '3.5': '3.0.3',  '3.6': '3.3.4',
        '3.7': '3.5.3',  '3.8': '',        '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'scikit-learn': {
        '2.7': '0.20.4', '3.5': '0.22.2', '3.6': '0.24.2',
        '3.7': '1.0.2',  '3.8': '',        '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'tensorflow': {
        '2.7': '1.14.0', '3.5': '1.14.0', '3.6': '2.6.5',
        '3.7': '2.10.1', '3.8': '2.13.1', '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'sqlalchemy': {
        '2.7': '1.3.24', '3.5': '1.3.24', '3.6': '1.4.51',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'redis': {
        '2.7': '3.5.3',  '3.5': '3.5.3',  '3.6': '4.5.5',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'pymongo': {
        '2.7': '3.12.3', '3.5': '3.12.3', '3.6': '3.13.0',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'celery': {
        '2.7': '4.4.7',  '3.5': '4.4.7',  '3.6': '5.2.7',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'scrapy': {
        '2.7': '1.8.2',  '3.5': '2.5.1',  '3.6': '2.6.3',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'beautifulsoup4': {
        '2.7': '4.9.3',  '3.5': '4.9.3',  '3.6': '',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'pillow': {
        '2.7': '6.2.2',  '3.5': '7.2.0',  '3.6': '8.4.0',
        '3.7': '9.5.0',  '3.8': '',        '3.9': '',
        '3.10': '',        '3.11': '',
    },
    'opencv-python': {
        '2.7': '4.2.0.32', '3.5': '4.2.0.32', '3.6': '4.5.5.64',
        '3.7': '4.7.0.72', '3.8': '',           '3.9': '',
    },
    'cryptography': {
        '2.7': '3.3.2',  '3.5': '3.3.2',  '3.6': '3.4.8',
        '3.7': '41.0.7', '3.8': '',        '3.9': '',
    },
    'pycryptodome': {
        '2.7': '3.15.0', '3.5': '3.15.0', '3.6': '3.19.1',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'paramiko': {
        '2.7': '2.12.0', '3.5': '2.12.0', '3.6': '3.4.1',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'lxml': {
        '2.7': '4.6.5',  '3.5': '4.6.5',  '3.6': '4.9.4',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'boto3': {
        '2.7': '1.17.112', '3.5': '1.17.112', '3.6': '1.26.165',
        '3.7': '',           '3.8': '',          '3.9': '',
    },
    'pyyaml': {
        '2.7': '5.4.1',  '3.5': '5.4.1',  '3.6': '',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'jinja2': {
        '2.7': '2.11.3', '3.5': '2.11.3', '3.6': '3.0.3',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'click': {
        '2.7': '7.1.2',  '3.5': '7.1.2',  '3.6': '8.0.4',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'twisted': {
        '2.7': '20.3.0', '3.5': '21.2.0', '3.6': '22.10.0',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'tornado': {
        '2.7': '5.1.1',  '3.5': '6.1',    '3.6': '6.1',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'gevent': {
        '2.7': '21.12.0', '3.5': '21.12.0', '3.6': '21.12.0',
        '3.7': '',          '3.8': '',         '3.9': '',
    },
    'aiohttp': {
        '3.6': '3.8.6',  '3.7': '3.9.1',  '3.8': '',
        '3.9': '',        '3.10': '',        '3.11': '',
    },
    'python-dateutil': {
        '2.7': '',        '3.5': '',        '3.6': '',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'six': {
        '2.7': '',        '3.5': '',        '3.6': '',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'pyzmq': {
        '2.7': '22.3.0', '3.5': '22.3.0', '3.6': '25.1.2',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'gitpython': {
        '2.7': '2.1.15', '3.5': '3.1.18', '3.6': '3.1.31',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'tweepy': {
        '2.7': '3.10.0', '3.5': '3.10.0', '3.6': '4.14.0',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'psycopg2-binary': {
        '2.7': '2.8.6',  '3.5': '2.8.6',  '3.6': '2.9.9',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'python-dotenv': {
        '2.7': '0.19.2', '3.5': '0.19.2', '3.6': '0.21.1',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'pyjwt': {
        '2.7': '1.7.1',  '3.5': '1.7.1',  '3.6': '2.8.0',
        '3.7': '',        '3.8': '',        '3.9': '',
    },
    'keras': {
        '2.7': '2.2.4',  '3.6': '2.3.1',  '3.7': '2.10.0',
        '3.8': '2.13.1', '3.9': '',
    },
    'nltk': {
        '2.7': '3.4.5',  '3.6': '3.6.7',  '3.7': '3.8.1',
        '3.8': '',        '3.9': '',
    },
    'networkx': {
        '2.7': '2.2',    '3.6': '2.6.3',  '3.7': '2.8.8',
        '3.8': '',        '3.9': '',
    },
}

# Extended import→package name mappings (superset of PLLM's module_link.json)
IMPORT_TO_PACKAGE = {
    'cv2':               'opencv-python',
    'sklearn':           'scikit-learn',
    'yaml':              'pyyaml',
    'PIL':               'Pillow',
    'pil':               'pillow',
    'bs4':               'beautifulsoup4',
    'serial':            'pyserial',
    'usb':               'pyusb',
    'gi':                'pygobject',
    'Crypto':            'pycryptodome',
    'crypto':            'pycryptodome',
    'cryptodome':        'pycryptodomex',
    'dateutil':          'python-dateutil',
    'dotenv':            'python-dotenv',
    'load_dotenv':       'python-dotenv',
    'jwt':               'PyJWT',
    'magic':             'python-magic',
    'psycopg2':          'psycopg2-binary',
    'attr':              'attrs',
    'skimage':           'scikit-image',
    'Bio':               'biopython',
    'bio':               'biopython',
    'docx':              'python-docx',
    'pptx':              'python-pptx',
    'git':               'gitpython',
    'memcache':          'python-memcached',
    'MySQLdb':           'mysqlclient',
    'mysqldb':           'mysqlclient',
    'zmq':               'pyzmq',
    'OpenSSL':           'pyopenssl',
    'openssl':           'pyopenssl',
    'bson':              'pymongo',
    # Django ecosystem
    'rest_framework':    'djangorestframework',
    'tastypie':          'django-tastypie',
    'guardian':          'django-guardian',
    'haystack':          'django-haystack',
    'debug_toolbar':     'django-debug-toolbar',
    'compressor':        'django-compressor',
    'storages':          'django-storages',
    'registration':      'django-registration',
    'imagekit':          'django-imagekit',
    'pipeline':          'django-pipeline',
    'social_auth':       'django-social-auth',
    'cms':               'django-cms',
    # From PLLM module_link.json
    'apiclient':         'google-api-python-client',
    'googleapiclient':   'google-api-python-client',
    'dns':               'dnspython3',
    'ffmpeg':            'python-ffmpeg',
    'github':            'pygithub',
    'jose':              'python-jose',
    'more_itertools':    'more-itertools',
    'multipart':         'python-multipart',
    'paho':              'paho-mqtt',
    'mosquitto':         'paho-mqtt',
    'chess':             'python-chess',
    'twitter':           'python-twitter',
    'visa':              'pyvisa',
    'xmpp':              'xmpppy',
    'socks':             'pysocks',
}


def get_compat_version(package: str, python_version: str) -> Optional[str]:
    """
    Look up a known-good version for package+python_version.

    Returns:
        A version string like '1.3.24' if a specific pin is needed,
        '' if latest is fine,
        None if this package is not in the compat map.
    """
    pkg_lower = package.lower()

    # Try exact key, then lowercase
    entry = COMPAT_MAP.get(package) or COMPAT_MAP.get(pkg_lower)
    if entry is None:
        return None  # Not in map — let LLM or PyPI handle it

    # Exact Python version match
    if python_version in entry:
        return entry[python_version]  # may be '' meaning "use latest"

    # Find closest version by numeric distance
    try:
        major, minor = python_version.split('.')
        py_num = float(f"{major}.{minor}")
    except ValueError:
        return ''

    best_val = ''
    best_dist = float('inf')
    for ver_key, ver_val in entry.items():
        try:
            k_major, k_minor = ver_key.split('.')
            k_num = float(f"{k_major}.{k_minor}")
            dist = abs(py_num - k_num)
            if dist < best_dist:
                best_dist = dist
                best_val = ver_val
        except ValueError:
            continue

    return best_val
