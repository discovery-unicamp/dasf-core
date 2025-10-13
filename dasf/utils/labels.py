#!/usr/bin/env python3
#
# from threading import Lock
#
# from dask.base import is_dask_collection
# from dask.core import get_dependencies, ishashable, istask
# from dask.dot import graphviz_to_file, to_graphviz
#
# inside_with = Lock()
#
# g_hash_attrs = dict()
# g_func_attrs = dict()
# g_data_attrs = dict()
#
#
# class DaskLabel(object):
#     def __init__(self, start, stop, label=None, color=None):
#         self.__label = label
#         self.__color = color
#         self.__start = start
#         self.__stop = stop
#         self.__hash_attrs = g_hash_attrs
#         self.__func_attrs = g_func_attrs
#         self.__data_attrs = g_data_attrs
#
#     def start(self, start):
#         self.__enter(start)
#
#     def stop(self, stop):
#         self.__exit(stop, None, None, None)
#
#     def __name(self, x):
#         try:
#             return str(hash(x))
#         except TypeError:
#             return str(hash(str(x)))
#
#     def __add_item(self, key, tag, label=None, color=None, atype="data"):
#         if not key in self.__data_attrs:
#             self.__hash_attrs[key] = dict()
#             # We use comment as a generic field for tag
#             self.__hash_attrs[key]["comment"] = tag
#             self.__hash_attrs[key]["xlabel"] = label
#             self.__hash_attrs[key]["color"] = color
#             self.__hash_attrs[key]["type"] = atype
#
#     def __add_func(self, key, tag, label, color):
#         if not key in self.__func_attrs:
#             self.__func_attrs[key] = dict()
#             # We use comment as a generic field for tag
#             self.__func_attrs[key]["comment"] = tag
#             if label:
#                 self.__func_attrs[key]["xlabel"] = label
#             if color:
#                 self.__func_attrs[key]["color"] = color
#                 self.__func_attrs[key]["style"] = "filled"
#
#     def __add_data(self, key, tag, label, color):
#         if not key in self.__data_attrs:
#             self.__data_attrs[key] = dict()
#             # We use comment as a generic field for tag
#             self.__data_attrs[key]["comment"] = tag
#             if label:
#                 self.__data_attrs[key]["xlabel"] = label
#             if color:
#                 self.__data_attrs[key]["color"] = color
#                 self.__data_attrs[key]["style"] = "filled"
#
#     def __generate_hashtable(self, data, delete_dup=False):
#         if not is_dask_collection(data):
#             raise Exception("This is not a Dask data: this is %s." % str(type(data)))
#         else:
#             dsk = data.dask
#
#         remove = set()
#
#         for k, v in dsk.items():
#             k_name = self.__name(k)
#             if istask(v):
#                 func_name = self.__name((k, "function"))
#
#                 if delete_dup and func_name in self.__hash_attrs:
#                     del self.__hash_attrs[func_name]
#                     remove.add(func_name)
#                 elif func_name not in remove:
#                     self.__add_item(
#                         func_name, k, self.__label, self.__color, atype="func"
#                     )
#
#                 for dep in get_dependencies(dsk, k):
#                     dep_name = self.__name(dep)
#                     if delete_dup and dep_name in self.__hash_attrs:
#                         del self.__hash_attrs[dep_name]
#                         remove.add(dep_name)
#                     elif dep_name not in remove:
#                         self.__add_item(dep_name, dep, self.__label, self.__color)
#
#     def __enter(self, dsk):
#         global inside_with
#
#         inside_with.acquire()
#
#         self.__generate_hashtable(dsk)
#
#         return self
#
#     def __enter__(self):
#         dsk = eval(self.__start)
#
#         return self.__enter(dsk)
#
#     def __exit(self, dsk, exc_type, exc_val, exc_tb):
#         global inside_with, g_hash_attrs, g_func_attrs, g_data_attrs
#
#         self.__generate_hashtable(dsk, delete_dup=True)
#
#         for k in self.__hash_attrs:
#             if self.__hash_attrs[k]["type"] == "data":
#                 self.__add_data(
#                     self.__hash_attrs[k]["comment"],
#                     k,
#                     self.__hash_attrs[k]["xlabel"],
#                     self.__hash_attrs[k]["color"],
#                 )
#             elif self.__hash_attrs[k]["type"] == "func":
#                 self.__add_func(
#                     self.__hash_attrs[k]["comment"],
#                     k,
#                     self.__hash_attrs[k]["xlabel"],
#                     self.__hash_attrs[k]["color"],
#                 )
#
#         g_hash_attrs = {**g_hash_attrs, **self.__hash_attrs}
#         g_func_attrs = {**g_func_attrs, **self.__func_attrs}
#         g_data_attrs = {**g_data_attrs, **self.__data_attrs}
#
#         inside_with.release()
#
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         dsk = eval(self.__stop)
#
#         return self.__exit(dsk, exc_type, exc_val, exc_tb)
#
#
# def get_attributes():
#     global inside_with, g_func_attrs, g_data_attrs
#
#     if inside_with.locked():
#         print("WARNING: it cannot reflect all attribute changes.")
#
#     return {"function_attributes": g_func_attrs, "data_attributes": g_data_attrs}
