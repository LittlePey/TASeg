import os
import re
import tempfile
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Tuple, Union
from petrel_client import client
import numpy as np
import io
import pickle
import torch

def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.
    Args:
        method (str): The method name to check.
        obj (object): The object to check.
    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.
    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    _allow_symlink = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def allow_symlink(self):
        return self._allow_symlink

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class PetrelBackend(BaseStorageBackend):
    """Petrel storage backend (for internal use).
    PetrelBackend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, PetrelBackend will read data
    from specified cluster or write data to it. Otherwise, PetrelBackend will
    access the default cluster.
    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Default: None.
        enable_mc (bool, optional): Whether to enable memcached support.
            Default: True.
    Examples:
        >>> filepath1 = 's3://path/of/file'
        >>> filepath2 = 'cluster-name:s3://path/of/file'
        >>> client = PetrelBackend()
        >>> client.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(self,
                 path_mapping: Optional[dict] = None, conf_path = '~/petreloss.conf'):
        
        self._client = client.Client(conf_path)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.
        Args:
            filepath (str): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.
        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.
        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def get(self, filepath: Union[str, Path], update_cache=False) -> memoryview:
        """Read data from a given ``filepath`` with 'rb' mode.
        Args:
            filepath (str or Path): Path to read data.
        Returns:
            memoryview: A memory view of expected bytes object to avoid
                copying. The memoryview object can be converted to bytes by
                ``value_buf.tobytes()``.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        value = self._client.Get(filepath, update_cache=update_cache)
        try:
            value_buf = memoryview(value)
        except:
            print('------------------', filepath)
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.
        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        Returns:
            str: Expected text reading from ``filepath``.
        """
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path], update_cache=True) -> None:
        """Save data to a given ``filepath``.
        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        if update_cache == True:
            try:
                self._client.put(filepath, obj, update_cache=update_cache)
            except:
                print("MC failed. Try direct put ", filepath)
                self._client.put(filepath, obj, update_cache=False)
        else:
            self._client.put(filepath, obj, update_cache=False)
        self._client.Get(filepath, update_cache=True)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Save data to a given ``filepath``.
        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Default: 'utf-8'.
        """
        self.put(bytes(obj, encoding=encoding), filepath)
        self._client.Get(filepath, update_cache=True)

    # numpy
    def save_np(self, filename, np_arr):
        with io.BytesIO() as f:
            np.save(f, np_arr)
            self.put(f.getvalue(), filename)

    def load_np(self, filename):
        return np.load(io.BytesIO(self.get(filename)))

    # binary
    def save_bin(self, filename, np_arr):
        self.put(np_arr.tostring(), filename)

    def load_bin(self, filename, dtype='float32'):
        if dtype=='float32':
            return np.frombuffer(self.get(filename), dtype=np.float32)
        elif dtype=='float64':
            return np.frombuffer(self.get(filename), dtype=np.float64)
        elif dtype=='long':
            return np.frombuffer(self.get(filename), dtype=np.long)
        elif dtype=='uint8':
            return np.frombuffer(self.get(filename), dtype=np.uint8)
        elif dtype=='uint32':
            return np.frombuffer(self.get(filename), dtype=np.uint32)

    # pickle
    def save_pkl(self, filename, pkl_arr_list):
        with io.BytesIO() as f:
            for pkl_arr in pkl_arr_list:
                pickle.dump(pkl_arr, f)
            self.put(f.getvalue(), filename)

    def load_pkl(self, filename, length):
        f = io.BytesIO(self.get(filename))
        pkl_arr_list = []
        for _ in range(length):
            pkl_arr = pickle.load(f)
            pkl_arr_list.append(pkl_arr)
        return pkl_arr_list

    # ckpt
    def save_ckpt(self, filename, checkpoint, _use_new_zipfile_serialization=None):
        with io.BytesIO() as f:
            if _use_new_zipfile_serialization is None:
                torch.save(checkpoint, f)
            else:
                torch.save(checkpoint, f, _use_new_zipfile_serialization=_use_new_zipfile_serialization)
            self.put(f.getvalue(), filename)

    def load_ckpt(self, filename, map_location=torch.device('cpu')):
        return torch.load(io.BytesIO(self.get(filename)), map_location=map_location)

    # image
    def save_img(self, filename, img):
        success, img_array = cv2.imencode('jpg', img)
        img_array_bytes = img_array.tostring()
        self.put(filename, img_array_bytes)

    def load_img(self, filename):
        img_mem_view = memoryview(self.get(filename))
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.
        Args:
            filepath (str or Path): Path to be removed.
        """
        if not has_method(self._client, 'delete'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `delete` method, please use a higher version or dev'
                ' branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        self._client.delete(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.
        Args:
            filepath (str or Path): Path to be checked whether exists.
        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        if not (has_method(self._client, 'contains')
                and has_method(self._client, 'isdir')):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `contains` and `isdir` methods, please use a higher'
                'version or dev branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.
        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.
        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        if not has_method(self._client, 'isdir'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `isdir` method, please use a higher version or dev'
                ' branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.
        Args:
            filepath (str or Path): Path to be checked whether it is a file.
        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        if not has_method(self._client, 'contains'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `contains` method, please use a higher version or '
                'dev branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        return self._client.contains(filepath)

    def join_path(self, filepath: Union[str, Path],
                  *filepaths: Union[str, Path]) -> str:
        """Concatenate all file paths.
        Args:
            filepath (str or Path): Path to be concatenated.
        Returns:
            str: The result after concatenation.
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith('/'):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_paths.append(self._format_path(self._map_path(path)))
        return '/'.join(formatted_paths)

    @contextmanager
    def get_local_path(
            self,
            filepath: Union[str,
                            Path]) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` and return a temporary path.
        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.
        Args:
            filepath (str | Path): Download a file from ``filepath``.
        Examples:
            >>> client = PetrelBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with client.get_local_path('s3://path/of/your/file') as path:
            ...     # do something here
        Yields:
            Iterable[str]: Only yield one temporary path.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.
        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.
        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.
        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.
        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        if not has_method(self._client, 'list'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `list` method, please use a higher version or dev'
                ' branch instead.')

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith('/'):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root):-1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(next_dir_path, list_dir,
                                                     list_file, suffix,
                                                     recursive)
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root):]
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)

    def list_dir_one_depth(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.
        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.
        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.
        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.
        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        if not has_method(self._client, 'list'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `list` method, please use a higher version or dev'
                ' branch instead.')

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            file_list = []
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                absolute_path = self.join_path(dir_path, path)
                rel_path = absolute_path[len(root):]
                if (suffix is None
                        or rel_path.endswith(suffix)) and list_file:
                    file_list.append(rel_path)
            return file_list
            
        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)

if __name__ == '__main__':
    backend = PetrelBackend()
