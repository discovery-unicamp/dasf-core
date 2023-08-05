#!/usr/bin/env python3

import os
import re
import nbformat
import unittest

from queue import Empty

try:
    from jupyter_client import KernelManager
except ImportError:
    print("FAILED: from IPython.kernel import KernelManager")
    from IPython.zmq.blockingkernelmanager import BlockingKernelManager as KernelManager

from pytest import fixture
from parameterized import parameterized_class

from dasf.utils.funcs import is_gpu_supported


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class CellExecutionException(Exception):
    """Raised when a ipynb kernel returns an error"""
    pass


class CellOutputException(Exception):
    """Raised when a ipynb kernel returns different output"""
    pass


def parameterize_ipynb():
    cur = os.getcwd()

    tutorials_dir = os.path.abspath(os.path.join(cur, "examples/tutorials/"))

    notebooks = []
    for root, dirs, files in os.walk(tutorials_dir, topdown=False):
        for name in files:
            if name.endswith(".ipynb"):
                notebooks.append({"notebook": os.path.join(root, name)})

    return notebooks


@unittest.skipIf(IN_GITHUB_ACTIONS, 'It is not working in CI/CD')
@parameterized_class(parameterize_ipynb())
class TestNotebooks(unittest.TestCase):
    def __sanitize(self, s):
        """Sanitize a string for comparison."""
        if not isinstance(s, str):
            return s
        # normalize newline:
        s = s.replace('\r\n', '\n')

        # ignore trailing newlines (but not space)
        s = s.rstrip('\n')

        # normalize hex addresses:
        s = re.sub(r'0x[a-f0-9]+', '0xFFFFFFFF', s)

        # normalize UUIDs:
        s = re.sub(r'[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}', 'U-U-I-D', s)

        return s

    def __compare_outputs(self, test, ref,
                          skip_compare=('png', 'traceback',
                                        'latex', 'prompt_number')):
        for key in ref:
            if key not in test:
                return False
            elif key not in skip_compare and \
                 self.__sanitize(test[key]) != self.__sanitize(ref[key]):
                return False
        return True

    def __run_cell(self, shell, iopub, cell, kc):
        kc.execute(cell.source)

        # wait for finish, maximum 20s
        while True:
            try:
                shell.get_msg(timeout=1)  # was 20
            except Empty:
                break
        
        outs = []

        while True:
            try:
                msg = iopub.get_msg(timeout=0.2)
            except Empty:
                break
            msg_type = msg['msg_type']
            if msg_type in ('status', 'execute_input'):
                continue
            elif msg_type == 'clear_output':
                outs = []
                continue

            content = msg['content']

            out = nbformat.NotebookNode(output_type=msg_type)

            if msg_type == 'stream':
                out.stream = content['name']
                out.text = content['text']
                out.data = content['text']
                out.name = content['name']
            elif msg_type in ('display_data', 'pyout', 'execute_result'):
                out['metadata'] = content['metadata']
                for mime, data in content['data'].items():
                    attr = mime.split('/')[-1].lower()
                    # this gets most right, but fix svg+html, plain
                    attr = attr.replace('+xml', '').replace('plain', 'text')
                    setattr(out, attr, data)
                out.data = content['data']

                if msg_type in ('execute_result', 'pyout'):
                    out.execution_count = content['execution_count']
            elif msg_type in ('pyerr', 'error'):
                out.ename = content['ename']
                out.evalue = content['evalue']
                out.traceback = content['traceback']
            else:
                print("unhandled iopub msg:", msg_type)

            outs.append(out)
        return outs

    def __test_notebook(self, nb):
        km = KernelManager()
        km.start_kernel(extra_arguments=['--pylab=inline'], stderr=open(os.devnull, 'w'))

        try:
            kc = km.client()
            kc.start_channels()
            iopub = kc.iopub_channel
        except AttributeError:
            print("AttributeError")
            # IPython 0.13
            kc = km
            kc.start_channels()
            iopub = kc.sub_channel

        shell = kc.shell_channel

        while True:
            try:
                iopub.get_msg(timeout=1)
            except Empty:
                break

        successes = 0
        failures = 0
        errors = 0

        ncell = 1
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue

            try:
                outs = self.__run_cell(shell, iopub, cell, kc)
                ncell += 1
            except Exception as e:
                error_str = f"failed to run cell {ncell}: " + str(e)

                kc.stop_channels()
                km.shutdown_kernel()
                del km

                raise CellExecutionException(error_str)

        if hasattr(cell, "outputs"):
            for out, ref in zip(outs, cell.outputs):
                if not self.__compare_outputs(out, ref):
                    kc.stop_channels()
                    km.shutdown_kernel()
                    del km

                    raise CellOutputException(out[:10] + "...")

        kc.stop_channels()
        km.shutdown_kernel()
        del km

    def test_notebook_execution(self):
        with open(self.notebook) as f:
            nb = nbformat.reads(f.read(), nbformat.current_nbformat)

            if 'test_requirements' in nb.metadata:
                test_reqs = nb.metadata['test_requirements']

                if 'single_gpu' in test_reqs and not is_gpu_supported():
                    self.skipTest("GPU is not available for testing.")

            self.__test_notebook(nb)
