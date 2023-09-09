import re
import time
from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from typing import Any, Callable
from warnings import warn


def flatten_hooks_dictionary(
    hooks: dict[str, dict | Callable], parent_name: str | None = None
):
    flattened_hooks = {}

    for key, value in hooks.items():
        name = f"{parent_name}:{key}" if parent_name else key

        if isinstance(value, dict):
            flattened_hooks.update(flatten_hooks_dictionary(value, name))
        elif callable(value):
            flattened_hooks[name] = value

    return flattened_hooks


def run_tasks_sequentially(tasks, handler):
    for task in tasks:
        handler(task)


def merge_hooks(*hooks: dict[str, Any]):
    merged_hooks = dict()

    for hook in hooks:
        flattened_hook = flatten_hooks_dictionary(hook)

        for key, value in flattened_hook.items():
            if merged_hooks[key]:
                merged_hooks[key].append(value)
            else:
                merged_hooks[key] = [value]

    for key in merged_hooks.keys():
        if len(merged_hooks[key]) > 1:
            hooks_list = merged_hooks[key]
            merged_hooks[key] = lambda *arguments: run_tasks_sequentially(
                hooks_list, lambda hook: hook(*arguments)
            )
        else:
            merged_hooks[key] = merged_hooks[key][0]

    return merged_hooks


def serial_caller(hooks: list[Callable], arguments: list[Any] = []):
    for hook in hooks:
        if len(signature(hook).parameters) > 0:
            hook(*arguments)
        else:
            hook()


def concurrent_caller(hooks: list[Callable], arguments: list[Any] = []):
    with ThreadPoolExecutor() as executor:
        executor.map(lambda hook: hook(*arguments), hooks)


def call_each_with(callbacks: list[Callable], argument: Any):
    for callback in callbacks:
        callback(argument)


class Hookable:
    def __init__(self):
        self._hooks = {}
        self._before = []
        self._after = []
        self._deprecated_messages = set()
        self._deprecated_hooks = {}

    def hook(self, name: str, function: Callable | None, options={}):
        if not name or not callable(function):
            return lambda: None

        original_name = name
        deprecated_hook = {}
        while self._deprecated_hooks.get(name):
            deprecated_hook = self._deprecated_hooks[name]
            name = deprecated_hook["to"]

        message = None
        if deprecated_hook and not options["allow_deprecated"]:
            message = deprecated_hook["message"]
            if not message:
                message = f"{original_name} hook has been deprecated" + (
                    f', please use {deprecated_hook["to"]}'
                    if deprecated_hook["to"]
                    else ""
                )

            if message not in self._deprecated_messages:
                warn(message)
                self._deprecated_messages.add(message)

        if function.__name__ == "<lambda>":
            function.__name__ = "_" + re.sub(r"\W+", "_", name) + "_hook_cb"

        self._hooks[name] = name in self._hooks or []
        self._hooks[name].append(function)

        def remove():
            nonlocal function
            if function:
                self.remove_hook(name, function)
                function = None

        return remove

    def hook_once(self, name: str, function: Callable):
        hook = None

        def run_once(*arguments):
            nonlocal hook
            if callable(hook):
                hook()

            hook = None
            return function(*arguments)

        hook = self.hook(name, run_once)
        return hook

    def remove_hook(self, name: str, function: Callable):
        if self._hooks[name]:
            if len(self._hooks[name]) == 0:
                del self._hooks[name]
            else:
                try:
                    index = self._hooks[name].index(function)
                    self._hooks[name][index:index] = []
                # if index is not found, ignore
                except ValueError:
                    pass

    def deprecate_hook(self, name: str, deprecated: Callable | str):
        self._deprecated_hooks[name] = (
            {"to": deprecated} if isinstance(deprecated, str) else deprecated
        )
        hooks = self._hooks[name] or []
        del self._hooks[name]
        for hook in hooks:
            self.hook(name, hook)

    def deprecate_hooks(self, deprecated_hooks: dict[str, Any]):
        self._deprecated_hooks.update(deprecated_hooks)
        for name in deprecated_hooks.keys():
            self.deprecate_hook(name, deprecated_hooks[name])

    def add_hooks(self, hooks: dict[str, Any]):
        hooks_to_be_added = flatten_hooks_dictionary(hooks)
        remove_fns = [self.hook(key, fn) for key, fn in hooks_to_be_added.items()]

        def function():
            for unreg in remove_fns:
                unreg()
            remove_fns[:] = []

        return function

    def remove_hooks(self, hooks: dict[str, Any]):
        hooks_to_be_removed = flatten_hooks_dictionary(hooks)
        for key, value in hooks_to_be_removed.items():
            self.remove_hook(key, value)

    def remove_all_hooks(self):
        for key in self._hooks.keys():
            del self._hooks[key]

    def call_hook(self, name: str, *arguments: Any):
        return self.call_hook_with(serial_caller, name, *arguments)

    def call_hook_concurrent(self, name: str, *arguments: Any):
        return self.call_hook_with(concurrent_caller, name, *arguments)

    def call_hook_with(self, caller: Callable, name: str, *arguments: Any):
        event = {"name": name, "args": arguments, "context": {}}

        call_each_with(self._before, event)

        result = caller(self._hooks[name] if name in self._hooks else [], arguments)

        call_each_with(self._after, event)

        return result

    def before_each(self, function: Callable):
        self._before.append(function)

        def remove_from_before_list():
            try:
                index = self._before.index(function)
                self._before[index:index] = []
            except ValueError:
                pass

        return remove_from_before_list

    def after_each(self, function: Callable):
        self._after.append(function)

        def remove_from_after_list():
            try:
                index = self._after.index(function)
                self._after[index:index] = []
            except ValueError:
                pass

        return remove_from_after_list


def create_hooks():
    return Hookable()


def starts_with_predicate(prefix: str):
    return lambda name: name.startswith(prefix)


def create_debugger(hooks: Hookable, _options: dict[str, Any] = {}):
    options = {"filter": lambda: True, **_options}

    predicate = options["filter"]
    if isinstance(predicate, str):
        predicate = starts_with_predicate(predicate)

    tag = f'[{options["tag"]}] ' if options["tag"] else ""
    start_times = {}

    def log_prefix(event: dict[str, Any]):
        return tag + event["name"] + "".ljust(int(event["id"]), "\0")

    id_control = {}

    def unsubscribe_before_each(event: dict[str, Any] | None = None):
        if event is None or not predicate(event["name"]):
            return

        id_control[event["name"]] = id_control.get(event.get("name")) or 0
        event["id"] = id_control[event["name"]]
        id_control[event["name"]] += 1
        start_times[log_prefix(event)] = time.time()

    unsubscribe_before = hooks.before_each(unsubscribe_before_each)

    def unsubscribe_after_each(event: dict[str, Any] | None = None):
        if event is None or not predicate(event["name"]):
            return

        label = log_prefix(event)
        elapsed_time = time.time() - start_times[label]
        print(f"Elapsed time for {label}: {elapsed_time} seconds")

        id_control[event["name"]] -= 1

    unsubscribe_after = hooks.after_each(unsubscribe_after_each)

    def stop_debbuging_and_remove_listeners():
        unsubscribe_before()
        unsubscribe_after()

    return {"close": lambda: stop_debbuging_and_remove_listeners()}
