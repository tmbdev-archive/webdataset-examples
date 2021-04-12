import os
import sys
import io
import pickle
import sqlite3
import time
import json as jsonlib

import torch

schema = """
create table if not exists log (
    step real,
    logtime real,
    key text,
    msg text,
    scalar real,
    json text,
    obj blob
)
"""


def torch_dumps(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getbuffer()


def torch_loads(buf):
    return torch.load(io.BytesIO(buf))


class Logger(object):
    def __init__(self, fname=None, sysinfo=True):
        if fname is None:
            import humanhash
            while True:
                fname = humanhash.uuid(words=2)[0]
                fname += ".sqlite3"
                if not os.path.exists(fname):
                    break
            print(f"log is {fname}", file=sys.stderr)
        self.con = sqlite3.connect(fname)
        self.con.execute(schema)
        self.last = 0
        self.interval = 10
        if sysinfo:
            self.sysinfo()

    def maybe_commit(self):
        if time.time() - self.last < self.interval:
            return
        for i in range(10):
            try:
                self.commit()
                break
            except sqlite3.OperationalError as exn:
                print("ERROR:", exn, file=sys.stderr)
                time.sleep(1.0)
        self.commit()
        self.last = time.time()

    def commit(self):
        self.con.commit()

    def flush(self):
        self.commit()

    def close(self):
        self.con.commit()
        self.con.close()

    def raw(
        self, key, step=None, msg=None, scalar=None, json=None, obj=None, walltime=None
    ):
        if msg is not None:
            assert isinstance(msg, (str, bytes)), msg
        if step is not None:
            step = float(step)
        if scalar is not None:
            scalar = float(scalar)
        if json is not None:
            assert isinstance(json, (str, bytes)), json
        # if obj is not None:
        #     assert isinstance(obj, bytes), obj
        if walltime is None:
            walltime = time.time()
        self.con.execute(
            "insert into log (logtime, step, key, msg, scalar, json, obj) "
            "values (?, ?, ?, ?, ?, ?, ?)",
            (walltime, step, key, msg, scalar, json, obj),
        )
        self.maybe_commit()

    def insert(
        self,
        key,
        step=None,
        msg=None,
        scalar=None,
        json=None,
        obj=None,
        dumps=pickle.dumps,
        walltime=None,
    ):
        if json is not None:
            json = jsonlib.dumps(json)
        if obj is not None:
            obj = dumps(obj)
        self.raw(
            key,
            step=step,
            msg=msg,
            scalar=scalar,
            json=json,
            obj=obj,
            walltime=walltime,
        )

    def scalar(self, key, scalar, step=None, **kw):
        self.insert(key, scalar=scalar, step=step, **kw)

    def message(self, key, msg, step=None, **kw):
        self.insert(key, msg=msg, step=step, **kw)

    def json(self, key, json, step=None, **kw):
        self.insert(key, json=json, step=step, **kw)

    def save(self, key, obj, step=None, **kw):
        self.insert(key, obj=obj, dumps=torch_dumps, step=step, **kw)

    def sysinfo(self):
        cmd = "hostname; uname"
        cmd += "; lsb_release -a"
        cmd += "; cat /proc/meminfo; cat /proc/cpuinfo"
        cmd += "; nvidia-smi -L"
        with os.popen(cmd) as stream:
            info = stream.read()
        self.message("__sysinfo__", info)

    # The following methods are for compatibility with Tensorboard

    def add_hparams(self, hparam_dict=None, metric_dict=None):
        if hparam_dict is not None:
            self.json("__hparams__", hparam_dict)
        if metric_dict is not None:
            self.json("__metrics__", metric_dict)

    def add_image(self, tag, obj, step=-1, walltime=None):
        # FIXME: convert to PNG
        self.save(tag, obj, step=step, walltime=walltime)

    def add_figure(self, tag, obj, step=-1, bins=None, walltime=None):
        # FIXME: convert to PNG
        self.save(tag, obj, step=step, walltime=walltime)

    def add_video(self, tag, obj, step=-1, bins=None, walltime=None):
        # FIXME: convert to MJPEG
        self.save(tag, obj, step=step, walltime=walltime)

    def add_audio(self, tag, obj, step=-1, bins=None, walltime=None):
        # FIXME: convert to FLAC
        self.save(tag, obj, step=step, walltime=walltime)

    def add_text(self, tag, obj, step=-1, bins=None, walltime=None):
        self.message(tag, obj, step=step, walltime=walltime)

    def add_embedding(self, tag, obj, step=-1, bins=None, walltime=None):
        raise Exception("unimplemented")

    def add_graph(self, tag, obj, step=-1, bins=None, walltime=None):
        raise Exception("unimplemented")

    def add_scalar(self, tag, value, step=-1, walltime=None):
        self.scalar(tag, value, step=step, walltime=walltime)

    def add_scalars(self, tag, value, step=-1, bins=None, walltime=None):
        for k, v in value.items():
            self.add_scalar(f"{tag}/{k}", v, step=step, walltime=walltime)

    def add_histogram(self, tag, values, step=-1, bins=None, walltime=None):
        raise Exception("unimplemented")

    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        raise Exception("unimplemented")

    def add_mesh(
        self,
        tag,
        vertices,
        colors=None,
        faces=None,
        config_dict=None,
        global_step=None,
        walltime=None,
    ):
        raise Exception("unimplemented")
