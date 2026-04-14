# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
from unittest.mock import patch

import pytest

_real_import = builtins.__import__


@pytest.mark.parametrize("algo", ["xxhash", "xxhash_cbor"])
def test_xxhash_missing_raises_import_error(algo: str):
    """CacheConfig must fail fast when xxhash is required but not installed."""
    from vllm.config.cache import CacheConfig

    def _mock_import(name, *args, **kwargs):
        if name == "xxhash":
            raise ModuleNotFoundError("No module named 'xxhash'",
                                      name="xxhash")
        return _real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        with pytest.raises(ImportError, match="xxhash"):
            CacheConfig(prefix_caching_hash_algo=algo)
