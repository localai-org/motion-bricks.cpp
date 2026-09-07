"""Deterministic GGUF identity/truncation rejection tests, not parser fuzzing.

Set MOTIONBRICKS_LIB and MOTIONBRICKS_SONIC_GGUF to trusted local artifacts.
Only throwaway copies are modified. The original model is never written.
"""
import ctypes as c
import os
from pathlib import Path
import shutil
import tempfile
import unittest


@unittest.skipUnless(os.getenv('MOTIONBRICKS_LIB') and os.getenv('MOTIONBRICKS_SONIC_GGUF'), 'local native SONIC artifacts required')
class Loading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib=c.CDLL(str(Path(os.environ['MOTIONBRICKS_LIB']).resolve()))
        cls.lib.mb_sonic_load.argtypes=[c.c_char_p,c.c_void_p,c.POINTER(c.c_void_p),c.c_void_p,c.c_uint64]
        cls.lib.mb_sonic_load.restype=c.c_uint32
        cls.lib.mb_sonic_free.argtypes=[c.c_void_p]
        cls.original=Path(os.environ['MOTIONBRICKS_SONIC_GGUF'])

    def reject(self,mutation):
        with tempfile.TemporaryDirectory(prefix='mb-sonic-rejection-') as directory:
            copy=Path(directory)/'model.gguf';shutil.copyfile(self.original,copy)
            with copy.open('r+b') as stream:mutation(stream)
            model=c.c_void_p();error=c.create_string_buffer(80)
            status=self.lib.mb_sonic_load(str(copy).encode(),None,c.byref(model),error,80)
            try:
                self.assertNotEqual(status,0);self.assertFalse(model.value)
                self.assertTrue(error.value);self.assertEqual(error.raw[-1],0)
            finally:self.lib.mb_sonic_free(model)

    def replace(self,before,after):
        def mutate(stream):
            header=stream.read(4096);offset=header.index(before)
            self.assertEqual(len(before),len(after))
            stream.seek(offset);stream.write(after)
        self.reject(mutate)

    def test_architecture(self):
        self.replace(b'g1-mode0-mlp-fsq32-v1',b'g1-mode1-mlp-fsq32-v1')

    def test_source_identity(self):
        self.replace(b'013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3',b'0'*64)

    def test_truncated_tensor(self):
        self.reject(lambda stream:stream.truncate(self.original.stat().st_size-4))

    def test_tiny_file(self):
        self.reject(lambda stream:stream.truncate(100))

if __name__=='__main__':unittest.main()
