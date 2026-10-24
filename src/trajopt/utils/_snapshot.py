"""Copy and freeze public data independently of the solver's mutable storage."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType
import copy
import numpy as np


class FrozenMapping(Mapping):
    """An immutable mapping with key and attribute access."""
    __slots__ = ('_values',)

    def __init__(self, values=(), **kwargs):
        object.__setattr__(self, '_values', MappingProxyType(dict(values, **kwargs)))

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise TypeError('Snapshot mappings are immutable')

    def __repr__(self):
        return f'FrozenMapping({dict(self)!r})'

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def readonly_array(value, *, dtype=None):
    """Copy an array to immutable storage that cannot be made writable."""
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError('Object arrays cannot be stored in immutable snapshots')
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


def freeze(value):
    """Recursively copy data and make arrays, mappings, and sequences immutable.

    Function references in the resolved configuration remain unchanged.
    Saved files do not support these references.
    """
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping((key, freeze(item)) for key, item in value.items())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) or hasattr(value, '__array__'):
        return readonly_array(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: freeze(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    return copy.deepcopy(value)


def thaw(value):
    """Return mutable copies of arrays and configuration mappings."""
    from trajopt.utils.tools import AttrDict
    if isinstance(value, Mapping):
        return AttrDict({key: thaw(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    if isinstance(value, frozenset):
        return {thaw(item) for item in value}
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: thaw(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    return copy.deepcopy(value)


def fingerprint(value):
    """Hash the configuration, including all numeric array contents.

    Mapping order does not affect the hash. Key types and sequence order do.
    Functions and other unsupported objects use their identity in the current
    process. Use this hash to check compatibility within that process.
    """
    import hashlib
    import struct
    from pathlib import Path

    digest = hashlib.sha256()

    def token(kind, payload=b''):
        kind = kind.encode('utf8')
        digest.update(struct.pack('!Q', len(kind)))
        digest.update(kind)
        digest.update(struct.pack('!Q', len(payload)))
        digest.update(payload)

    def visit(item):
        if item is None:
            token('none')
        elif isinstance(item, bool):
            token('bool', b'1' if item else b'0')
        elif isinstance(item, np.generic):
            visit(np.asarray(item))
        elif isinstance(item, int):
            token('int', str(item).encode('ascii'))
        elif isinstance(item, float):
            token('float', struct.pack('!d', item))
        elif isinstance(item, complex):
            token('complex', struct.pack('!dd', item.real, item.imag))
        elif isinstance(item, str):
            token('str', item.encode('utf8'))
        elif isinstance(item, bytes):
            token('bytes', item)
        elif isinstance(item, np.ndarray) or hasattr(item, '__array__'):
            array = np.asarray(item)
            if array.dtype.hasobject:
                raise TypeError('Object arrays cannot be fingerprinted as numeric configuration')
            token('array')
            visit(array.dtype.str)
            if array.dtype.fields is not None:
                visit(array.dtype.descr)
            visit(array.shape)
            token('contents', array.tobytes(order='C'))
        elif isinstance(item, Mapping):
            token('mapping', str(len(item)).encode('ascii'))
            for key in sorted(item, key=fingerprint):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            token('tuple' if isinstance(item, tuple) else 'list', str(len(item)).encode('ascii'))
            for child in item:
                visit(child)
        elif isinstance(item, (set, frozenset)):
            token('set', str(len(item)).encode('ascii'))
            for child in sorted(item, key=fingerprint):
                visit(child)
        elif isinstance(item, Path):
            token('path', str(item).encode('utf8'))
        elif is_dataclass(item) and not isinstance(item, type):
            token('dataclass', f'{type(item).__module__}.{type(item).__qualname__}'.encode('utf8'))
            visit({field.name: getattr(item, field.name) for field in fields(item) if field.init})
        else:
            token('opaque', f'{type(item).__module__}.{type(item).__qualname__}:{id(item)}'.encode('utf8'))

    visit(value)
    return digest.hexdigest()
