"""Build the curated bugfix suite.

Each task is a self-contained Python module with one real bug, a test that
fails on the bug and passes on the fix, and (usually) a guard test that must
keep passing.  Tasks are deliberately small so the benchmark stays fast and
deterministic: the point is to exercise the control plane, not to hide a bug
inside a large program.

    python3 -m eval.suites.curated.build           # write suite.json
    python3 -m eval.suites.curated.build --check   # verify every task first

``--check`` runs each task's tests against the buggy and fixed module in a
throwaway directory and fails loudly if the bug is not real or the fix is not
complete.  Run it after editing a task; suite.json is the generated artifact.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from textwrap import dedent

import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "suite.json")
HARD_OUT = os.path.join(os.path.dirname(HERE), "hard", "suite.json")
MULTI_OUT = os.path.join(os.path.dirname(HERE), "multifile", "suite.json")
DISTRACTORS = 3


def T(id: str, task: str, buggy: str, fixed: str, test: str) -> dict:
    mod = f"{id}.py"
    test_file = f"test_{id}.py"
    keep = [f"{test_file}::test_guard"] if "def test_guard" in test else []
    return {
        "id": id,
        "task": task,
        "expected_files": [mod],
        "files": {mod: dedent(buggy).strip() + "\n", test_file: dedent(test).strip() + "\n"},
        "fix": {mod: dedent(fixed).strip() + "\n"},
        "expect_pass": [f"{test_file}::test_main"],
        "must_keep_passing": keep,
    }


TASKS = [
    T("c01_slice_off_by_one",
      "fix first_n so it returns exactly the first n items",
      """\
      def first_n(xs, n):
          return xs[:n - 1]
      """,
      """\
      def first_n(xs, n):
          return xs[:n]
      """,
      """\
      from c01_slice_off_by_one import first_n

      def test_main():
          assert first_n([1, 2, 3, 4], 2) == [1, 2]

      def test_guard():
          assert first_n([1, 2, 3], 10) == [1, 2, 3]
      """),

    T("c02_inclusive_upper",
      "fix in_range so an upper bound is inclusive",
      """\
      def in_range(x, lo, hi):
          return lo <= x < hi
      """,
      """\
      def in_range(x, lo, hi):
          return lo <= x <= hi
      """,
      """\
      from c02_inclusive_upper import in_range

      def test_main():
          assert in_range(5, 0, 5) is True

      def test_guard():
          assert in_range(6, 0, 5) is False
      """),

    T("c03_empty_first",
      "fix first_or_none so it handles an empty list",
      """\
      def first_or_none(xs):
          return xs[0]
      """,
      """\
      def first_or_none(xs):
          return xs[0] if xs else None
      """,
      """\
      from c03_empty_first import first_or_none

      def test_main():
          assert first_or_none([]) is None

      def test_guard():
          assert first_or_none([7]) == 7
      """),

    T("c04_mutable_default",
      "fix add so repeated calls do not share one default list",
      """\
      def add(item, acc=[]):
          acc.append(item)
          return acc
      """,
      """\
      def add(item, acc=None):
          acc = [] if acc is None else acc
          acc.append(item)
          return acc
      """,
      """\
      from c04_mutable_default import add

      def test_main():
          assert add(1) == [1]
          assert add(2) == [2]

      def test_guard():
          assert add(2, [1]) == [1, 2]
      """),

    T("c05_missing_key",
      "fix lookup so a missing key returns None instead of raising",
      """\
      def lookup(d, k):
          return d[k]
      """,
      """\
      def lookup(d, k):
          return d.get(k)
      """,
      """\
      from c05_missing_key import lookup

      def test_main():
          assert lookup({}, "x") is None

      def test_guard():
          assert lookup({"x": 3}, "x") == 3
      """),

    T("c06_divide_zero",
      "fix safe_div so dividing by zero returns 0",
      """\
      def safe_div(a, b):
          return a / b
      """,
      """\
      def safe_div(a, b):
          return a / b if b else 0
      """,
      """\
      from c06_divide_zero import safe_div

      def test_main():
          assert safe_div(3, 0) == 0

      def test_guard():
          assert safe_div(6, 3) == 2
      """),

    T("c07_int_average",
      "fix average so it keeps the fractional part",
      """\
      def average(a, b):
          return (a + b) // 2
      """,
      """\
      def average(a, b):
          return (a + b) / 2
      """,
      """\
      from c07_int_average import average

      def test_main():
          assert average(2, 3) == 2.5

      def test_guard():
          assert average(2, 4) == 3.0
      """),

    T("c08_float_equality",
      "fix nearly_equal so float rounding is tolerated",
      """\
      def nearly_equal(a, b):
          return a == b
      """,
      """\
      def nearly_equal(a, b):
          return abs(a - b) < 1e-9
      """,
      """\
      from c08_float_equality import nearly_equal

      def test_main():
          assert nearly_equal(0.1 + 0.2, 0.3) is True

      def test_guard():
          assert nearly_equal(1.0, 1.0) is True
      """),

    T("c09_round_number",
      "fix round_number so 2.6 rounds to 3",
      """\
      def round_number(x):
          return int(x)
      """,
      """\
      def round_number(x):
          return round(x)
      """,
      """\
      from c09_round_number import round_number

      def test_main():
          assert round_number(2.6) == 3

      def test_guard():
          assert round_number(2.4) == 2
      """),

    T("c10_split_empty",
      "fix split_csv so an empty string yields no fields",
      """\
      def split_csv(s):
          return s.split(",")
      """,
      """\
      def split_csv(s):
          return s.split(",") if s else []
      """,
      """\
      from c10_split_empty import split_csv

      def test_main():
          assert split_csv("") == []

      def test_guard():
          assert split_csv("a,b") == ["a", "b"]
      """),

    T("c11_join_numbers",
      "fix join_numbers so integers can be joined",
      """\
      def join_numbers(nums):
          return ",".join(nums)
      """,
      """\
      def join_numbers(nums):
          return ",".join(map(str, nums))
      """,
      """\
      from c11_join_numbers import join_numbers

      def test_main():
          assert join_numbers([1, 2]) == "1,2"

      def test_guard():
          assert join_numbers(["a", "b"]) == "a,b"
      """),

    T("c12_percent_format",
      "fix as_percent so 0.5 renders as 50%",
      """\
      def as_percent(x):
          return f"{x}%"
      """,
      """\
      def as_percent(x):
          return f"{x:.0%}"
      """,
      """\
      from c12_percent_format import as_percent

      def test_main():
          assert as_percent(0.5) == "50%"

      """),

    T("c13_regex_literal_dot",
      "fix is_dot so the dot is matched literally",
      """\
      import re

      def is_dot(s):
          return bool(re.fullmatch(".", s))
      """,
      """\
      import re

      def is_dot(s):
          return bool(re.fullmatch(r"\\.", s))
      """,
      """\
      from c13_regex_literal_dot import is_dot

      def test_main():
          assert is_dot(".") is True
          assert is_dot("a") is False

      def test_guard():
          assert is_dot(".") is True
      """),

    T("c14_dict_delete_iter",
      "fix clear_dict so it can delete while iterating",
      """\
      def clear_dict(d):
          for k in d:
              del d[k]
          return d
      """,
      """\
      def clear_dict(d):
          for k in list(d):
              del d[k]
          return d
      """,
      """\
      from c14_dict_delete_iter import clear_dict

      def test_main():
          assert clear_dict({"a": 1, "b": 2}) == {}

      def test_guard():
          assert clear_dict({}) == {}
      """),

    T("c15_sort_desc",
      "fix sort_desc so it returns items in descending order",
      """\
      def sort_desc(xs):
          return sorted(xs)
      """,
      """\
      def sort_desc(xs):
          return sorted(xs, reverse=True)
      """,
      """\
      from c15_sort_desc import sort_desc

      def test_main():
          assert sort_desc([3, 1, 2]) == [3, 2, 1]

      def test_guard():
          assert sort_desc([]) == []
      """),

    T("c16_unique_order",
      "fix unique so it preserves first-seen order",
      """\
      def unique(xs):
          return list(set(xs))
      """,
      """\
      def unique(xs):
          return list(dict.fromkeys(xs))
      """,
      """\
      from c16_unique_order import unique

      def test_main():
          assert unique([3, 1, 3, 2]) == [3, 1, 2]

      def test_guard():
          assert unique([]) == []
      """),

    T("c17_flatten_one",
      "fix flatten so nested lists are flattened one level",
      """\
      def flatten(xss):
          return [x for x in xss]
      """,
      """\
      def flatten(xss):
          return [x for sub in xss for x in sub]
      """,
      """\
      from c17_flatten_one import flatten

      def test_main():
          assert flatten([[1, 2], [3]]) == [1, 2, 3]

      def test_guard():
          assert flatten([]) == []
      """),

    T("c18_factorial_base",
      "fix factorial so factorial(0) is 1",
      """\
      def factorial(n):
          if n == 0:
              return 0
          return n * factorial(n - 1)
      """,
      """\
      def factorial(n):
          if n == 0:
              return 1
          return n * factorial(n - 1)
      """,
      """\
      from c18_factorial_base import factorial

      def test_main():
          assert factorial(0) == 1

      """),

    T("c19_fib_child",
      "fix fib so it recurses on both preceding terms",
      """\
      def fib(n):
          if n < 2:
              return n
          return fib(n - 1) + fib(n - 1)
      """,
      """\
      def fib(n):
          if n < 2:
              return n
          return fib(n - 1) + fib(n - 2)
      """,
      """\
      from c19_fib_child import fib

      def test_main():
          assert fib(5) == 5

      def test_guard():
          assert fib(1) == 1
      """),

    T("c20_numeric_strings",
      "fix add_strings so numeric strings are added numerically",
      """\
      def add_strings(a, b):
          return a + b
      """,
      """\
      def add_strings(a, b):
          return int(a) + int(b)
      """,
      """\
      from c20_numeric_strings import add_strings

      def test_main():
          assert add_strings("2", "3") == 5

      """),

    T("c21_truthiness",
      "fix truthy so any truthy value counts, not just True",
      """\
      def truthy(x):
          if x == True:
              return True
          return False
      """,
      """\
      def truthy(x):
          return bool(x)
      """,
      """\
      from c21_truthiness import truthy

      def test_main():
          assert truthy("yes") is True

      def test_guard():
          assert truthy(0) is False
      """),

    T("c22_casefold",
      "fix normalize_case so German sharp s folds to ss",
      """\
      def normalize_case(s):
          return s.lower()
      """,
      """\
      def normalize_case(s):
          return s.casefold()
      """,
      """\
      from c22_casefold import normalize_case

      def test_main():
          assert normalize_case("ß") == "ss"

      def test_guard():
          assert normalize_case("ABC") == "abc"
      """),

    T("c23_path_absolute_override",
      "fix join_under so an absolute child cannot escape the base",
      """\
      import os

      def join_under(base, name):
          return os.path.join(base, name)
      """,
      """\
      import os

      def join_under(base, name):
          return os.path.join(base, name.lstrip("/"))
      """,
      """\
      from c23_path_absolute_override import join_under

      def test_main():
          assert join_under("/srv/data", "/etc/passwd") == "/srv/data/etc/passwd"

      def test_guard():
          assert join_under("/srv/data", "x") == "/srv/data/x"
      """),

    T("c24_trailing_slash",
      "fix basename so a trailing slash is ignored",
      """\
      def basename(p):
          return p.split("/")[-1]
      """,
      """\
      def basename(p):
          return p.rstrip("/").split("/")[-1]
      """,
      """\
      from c24_trailing_slash import basename

      def test_main():
          assert basename("a/b/") == "b"

      def test_guard():
          assert basename("a/b") == "b"
      """),

    T("c25_json_set",
      "fix to_json so sets are serialisable",
      """\
      import json

      def to_json(obj):
          return json.dumps(obj)
      """,
      """\
      import json

      def to_json(obj):
          return json.dumps(obj, default=list)
      """,
      """\
      import json
      from c25_json_set import to_json

      def test_main():
          assert json.loads(to_json({"s": {1}})) == {"s": [1]}

      def test_guard():
          assert json.loads(to_json({"a": 1})) == {"a": 1}
      """),

    T("c26_alias_copy",
      "fix append_copy so it does not mutate the input list",
      """\
      def append_copy(xs, item):
          out = xs
          out.append(item)
          return out
      """,
      """\
      def append_copy(xs, item):
          out = list(xs)
          out.append(item)
          return out
      """,
      """\
      from c26_alias_copy import append_copy

      def test_main():
          xs = [1]
          assert append_copy(xs, 2) == [1, 2]
          assert xs == [1]

      def test_guard():
          assert append_copy([], 1) == [1]
      """),

    T("c27_generator_exhaust",
      "fix twice_sum so a generator can be summed twice",
      """\
      def twice_sum(items):
          return sum(items) + sum(items)
      """,
      """\
      def twice_sum(items):
          vals = list(items)
          return sum(vals) + sum(vals)
      """,
      """\
      from c27_generator_exhaust import twice_sum

      def test_main():
          assert twice_sum(iter([1, 2])) == 6

      def test_guard():
          assert twice_sum([1]) == 2
      """),

    T("c28_enumerate_start",
      "fix label_items so numbering starts at one",
      """\
      def label_items(items):
          return [(i, x) for i, x in enumerate(items)]
      """,
      """\
      def label_items(items):
          return [(i, x) for i, x in enumerate(items, 1)]
      """,
      """\
      from c28_enumerate_start import label_items

      def test_main():
          assert label_items(["a"]) == [(1, "a")]

      def test_guard():
          assert label_items([]) == []
      """),

    T("c29_zip_pad",
      "fix zip_pad so unequal lengths are padded with zero",
      """\
      def zip_pad(a, b):
          return list(zip(a, b))
      """,
      """\
      from itertools import zip_longest

      def zip_pad(a, b):
          return list(zip_longest(a, b, fillvalue=0))
      """,
      """\
      from c29_zip_pad import zip_pad

      def test_main():
          assert zip_pad([1, 2], [3]) == [(1, 3), (2, 0)]

      def test_guard():
          assert zip_pad([], []) == []
      """),

    T("c30_all_empty",
      "fix all_positive so an empty list is not reported as all positive",
      """\
      def all_positive(xs):
          return all(x > 0 for x in xs)
      """,
      """\
      def all_positive(xs):
          return bool(xs) and all(x > 0 for x in xs)
      """,
      """\
      from c30_all_empty import all_positive

      def test_main():
          assert all_positive([]) is False

      def test_guard():
          assert all_positive([1, 2]) is True
      """),

    T("c31_max_empty",
      "fix max_or_none so an empty list returns None",
      """\
      def max_or_none(xs):
          return max(xs)
      """,
      """\
      def max_or_none(xs):
          return max(xs, default=None)
      """,
      """\
      from c31_max_empty import max_or_none

      def test_main():
          assert max_or_none([]) is None

      def test_guard():
          assert max_or_none([1, 3, 2]) == 3
      """),

    T("c32_int_from_float_str",
      "fix to_int so a float string is truncated toward zero",
      """\
      def to_int(s):
          return int(s)
      """,
      """\
      def to_int(s):
          return int(float(s))
      """,
      """\
      from c32_int_from_float_str import to_int

      def test_main():
          assert to_int("4.2") == 4

      def test_guard():
          assert to_int("7") == 7
      """),

    T("c33_endswith_suffix",
      "fix ends_with so it tests the suffix, not the prefix",
      """\
      def ends_with(s, suffix):
          return s[:len(suffix)] == suffix
      """,
      """\
      def ends_with(s, suffix):
          return s.endswith(suffix)
      """,
      """\
      from c33_endswith_suffix import ends_with

      def test_main():
          assert ends_with("sing", "ing") is True
          assert ends_with("ings", "ing") is False

      def test_guard():
          assert ends_with("x", "y") is False
      """),

    T("c34_replace_all",
      "fix replace_all so every occurrence is replaced",
      """\
      def replace_all(s, old, new):
          return s.replace(old, new, 1)
      """,
      """\
      def replace_all(s, old, new):
          return s.replace(old, new)
      """,
      """\
      from c34_replace_all import replace_all

      def test_main():
          assert replace_all("aaa", "a", "b") == "bbb"

      def test_guard():
          assert replace_all("abc", "x", "y") == "abc"
      """),

    T("c35_count_overlap",
      "fix count_overlap so overlapping occurrences are counted",
      """\
      def count_overlap(s, sub):
          return s.count(sub)
      """,
      """\
      def count_overlap(s, sub):
          n = 0
          for i in range(len(s) - len(sub) + 1):
              if s[i:i + len(sub)] == sub:
                  n += 1
          return n
      """,
      """\
      from c35_count_overlap import count_overlap

      def test_main():
          assert count_overlap("aaa", "aa") == 2

      def test_guard():
          assert count_overlap("abc", "x") == 0
      """),

    T("c36_title_words",
      "fix title_words so every word is capitalised",
      """\
      def title_words(s):
          return s.capitalize()
      """,
      """\
      def title_words(s):
          return s.title()
      """,
      """\
      from c36_title_words import title_words

      def test_main():
          assert title_words("hello world") == "Hello World"

      def test_guard():
          assert title_words("x") == "X"
      """),

    T("c37_year_4digit",
      "fix year so it formats four digits",
      """\
      import datetime

      def year(d):
          return d.strftime("%y")
      """,
      """\
      import datetime

      def year(d):
          return d.strftime("%Y")
      """,
      """\
      import datetime
      from c37_year_4digit import year

      def test_main():
          assert year(datetime.date(2024, 1, 1)) == "2024"

      """),

    T("c38_timedelta_days",
      "fix days_span so multi-day spans report the day count",
      """\
      import datetime

      def days_span(td):
          return td.seconds // 86400
      """,
      """\
      import datetime

      def days_span(td):
          return td.days
      """,
      """\
      import datetime
      from c38_timedelta_days import days_span

      def test_main():
          assert days_span(datetime.timedelta(days=2)) == 2

      def test_guard():
          assert days_span(datetime.timedelta(days=0)) == 0
      """),

    T("c39_clamp_args",
      "fix clamp so it bounds a value between lo and hi",
      """\
      def clamp(v, lo, hi):
          return min(lo, max(hi, v))
      """,
      """\
      def clamp(v, lo, hi):
          return min(hi, max(lo, v))
      """,
      """\
      from c39_clamp_args import clamp

      def test_main():
          assert clamp(5, 0, 3) == 3

      def test_guard():
          assert clamp(-1, 0, 3) == 0
      """),

    T("c40_chunk_last",
      "fix chunk so the final partial chunk is included",
      """\
      def chunk(xs, n):
          return [xs[i:i + n] for i in range(0, len(xs) - n + 1, n)]
      """,
      """\
      def chunk(xs, n):
          return [xs[i:i + n] for i in range(0, len(xs), n)]
      """,
      """\
      from c40_chunk_last import chunk

      def test_main():
          assert chunk([1, 2, 3], 2) == [[1, 2], [3]]

      def test_guard():
          assert chunk([], 2) == []
      """),

    T("c41_dedupe_adjacent",
      "fix dedupe_adjacent so only consecutive duplicates are removed",
      """\
      def dedupe_adjacent(xs):
          return list(set(xs))
      """,
      """\
      from itertools import groupby

      def dedupe_adjacent(xs):
          return [x for x, _ in groupby(xs)]
      """,
      """\
      from c41_dedupe_adjacent import dedupe_adjacent

      def test_main():
          assert dedupe_adjacent([1, 1, 2, 1]) == [1, 2, 1]

      def test_guard():
          assert dedupe_adjacent([]) == []
      """),

    T("c42_merge_no_mutate",
      "fix merge so neither input dict is modified",
      """\
      def merge(a, b):
          a.update(b)
          return a
      """,
      """\
      def merge(a, b):
          return {**a, **b}
      """,
      """\
      from c42_merge_no_mutate import merge

      def test_main():
          a = {"x": 1}
          assert merge(a, {"y": 2}) == {"x": 1, "y": 2}
          assert a == {"x": 1}

      def test_guard():
          assert merge({}, {}) == {}
      """),

    T("c43_counter_key",
      "fix bump so counting an unseen key starts from zero",
      """\
      def bump(d, k):
          d[k] += 1
          return d
      """,
      """\
      def bump(d, k):
          d[k] = d.get(k, 0) + 1
          return d
      """,
      """\
      from c43_counter_key import bump

      def test_main():
          assert bump({}, "x") == {"x": 1}

      def test_guard():
          assert bump({"x": 2}, "x") == {"x": 3}
      """),

    T("c44_exception_reraise",
      "fix parse so a bad value raises instead of being swallowed",
      """\
      def parse(s):
          try:
              return int(s)
          except ValueError:
              return None
      """,
      """\
      def parse(s):
          return int(s)
      """,
      """\
      import pytest
      from c44_exception_reraise import parse

      def test_main():
          with pytest.raises(ValueError):
              parse("nope")

      def test_guard():
          assert parse("4") == 4
      """),

    T("c45_finally_return",
      "fix value so a finally block does not override the return",
      """\
      def value():
          try:
              return 1
          finally:
              return 2
      """,
      """\
      def value():
          try:
              return 1
          finally:
              pass
      """,
      """\
      from c45_finally_return import value

      def test_main():
          assert value() == 1

      """),

    T("c46_is_compare",
      "fix same_text so equal strings compare equal, not by identity",
      """\
      def same_text(a, b):
          return a is b
      """,
      """\
      def same_text(a, b):
          return a == b
      """,
      """\
      from c46_is_compare import same_text

      def test_main():
          assert same_text("".join(["a", "b"]), "ab") is True

      def test_guard():
          assert same_text("a", "b") is False
      """),

    T("c47_negative_index",
      "fix last so it returns the final element",
      """\
      def last(xs):
          return xs[len(xs)]
      """,
      """\
      def last(xs):
          return xs[len(xs) - 1]
      """,
      """\
      from c47_negative_index import last

      def test_main():
          assert last([1, 2, 3]) == 3

      """),

    T("c48_string_reverse",
      "fix reverse so it reverses the whole string",
      """\
      def reverse(s):
          return s[::-2]
      """,
      """\
      def reverse(s):
          return s[::-1]
      """,
      """\
      from c48_string_reverse import reverse

      def test_main():
          assert reverse("abcd") == "dcba"

      def test_guard():
          assert reverse("") == ""
      """),

    T("c49_sorted_key_int",
      "fix sort_numbers so numbers are compared numerically",
      """\
      def sort_numbers(xs):
          return sorted(xs, key=str)
      """,
      """\
      def sort_numbers(xs):
          return sorted(xs, key=int)
      """,
      """\
      from c49_sorted_key_int import sort_numbers

      def test_main():
          assert sort_numbers(["10", "2"]) == ["2", "10"]

      def test_guard():
          assert sort_numbers([]) == []
      """),

    T("c50_percentage_calc",
      "fix percent so a part of a total is scaled to 100",
      """\
      def percent(part, total):
          return part / total
      """,
      """\
      def percent(part, total):
          return part / total * 100
      """,
      """\
      from c50_percentage_calc import percent

      def test_main():
          assert percent(1, 4) == 25.0

      """),
]


def M(id: str, task: str, modules: dict, fix: dict, tests: dict, expected: list) -> dict:
    """A task whose fix spans more than one module.

    The tests import only the public module, but the change cannot land
    without touching ``expected``: retrieval has to surface all of them and
    the plan has to commit to editing each one.
    """
    files = dict(modules)
    files.update(tests)
    return {
        "id": id,
        "task": task,
        "expected_files": list(expected),
        "files": {k: dedent(v).strip() + "\n" for k, v in files.items()},
        "fix": {k: dedent(v).strip() + "\n" for k, v in fix.items()},
        "expect_pass": [f"{name}::test_main" for name in tests],
        "must_keep_passing": [f"{name}::test_guard" for name, text in tests.items()
                              if "def test_guard" in text],
    }


MULTI = [
    M("m01_rename_total",
      "fix the off-by-one in total, rename it to sum_prices in m01_core, and update the m01_cart caller",
      {"m01_core.py": """
        def total(prices):
            return sum(prices) - 1
      """,
       "m01_cart.py": """
        from m01_core import total

        def checkout(prices):
            return total(prices)
      """},
      {"m01_core.py": """
        def sum_prices(prices):
            return sum(prices)
      """,
       "m01_cart.py": """
        from m01_core import sum_prices

        def checkout(prices):
            return sum_prices(prices)
      """},
      {"test_m01.py": """
        from m01_core import sum_prices
        from m01_cart import checkout

        def test_main():
            assert sum_prices([2, 3]) == 5
            assert checkout([2, 3]) == 5
      """},
      ["m01_core.py", "m01_cart.py"]),

    M("m02_suffix_param",
      "add a suffix parameter to money in m02_fmt and pass it from m02_invoice",
      {"m02_fmt.py": """
        def money(value):
            return f"{value:.2f}"
      """,
       "m02_invoice.py": """
        from m02_fmt import money

        def line(value):
            return money(value)
      """},
      {"m02_fmt.py": """
        def money(value, suffix=""):
            return f"{value:.2f}{suffix}"
      """,
       "m02_invoice.py": """
        from m02_fmt import money

        def line(value):
            return money(value, " USD")
      """},
      {"test_m02.py": """
        from m02_invoice import line

        def test_main():
            assert line(2) == "2.00 USD"
      """},
      ["m02_fmt.py", "m02_invoice.py"]),

    M("m03_rename_area",
      "fix the multiply in area, rename it to rect_area in m03_geom, and update m03_shapes",
      {"m03_geom.py": """
        def area(w, h):
            return w + h
      """,
       "m03_shapes.py": """
        from m03_geom import area

        def square_area(side):
            return area(side, side)
      """},
      {"m03_geom.py": """
        def rect_area(w, h):
            return w * h
      """,
       "m03_shapes.py": """
        from m03_geom import rect_area

        def square_area(side):
            return rect_area(side, side)
      """},
      {"test_m03.py": """
        from m03_geom import rect_area
        from m03_shapes import square_area

        def test_main():
            assert rect_area(2, 3) == 6
            assert square_area(3) == 9
      """},
      ["m03_geom.py", "m03_shapes.py"]),

    M("m04_strict_param",
      "make is_name reject non-identifiers via a strict flag in m04_validate and use it from m04_form",
      {"m04_validate.py": """
        def is_name(value):
            return bool(value)
      """,
       "m04_form.py": """
        from m04_validate import is_name

        def field(value):
            return is_name(value)
      """},
      {"m04_validate.py": """
        def is_name(value, strict=False):
            return value.isidentifier() if strict else bool(value)
      """,
       "m04_form.py": """
        from m04_validate import is_name

        def field(value):
            return is_name(value, strict=True)
      """},
      {"test_m04.py": """
        from m04_form import field

        def test_main():
            assert field("1bad") is False
            assert field("good") is True
      """},
      ["m04_validate.py", "m04_form.py"]),

    M("m05_rename_label",
      "make label return a string, rename it to title in m05_tag, and update m05_doc",
      {"m05_tag.py": """
        def label(x):
            return x
      """,
       "m05_doc.py": """
        from m05_tag import label

        def heading(x):
            return "# " + label(x)
      """},
      {"m05_tag.py": """
        def title(x):
            return str(x)
      """,
       "m05_doc.py": """
        from m05_tag import title

        def heading(x):
            return "# " + title(x)
      """},
      {"test_m05.py": """
        from m05_tag import title
        from m05_doc import heading

        def test_main():
            assert title(3) == "3"
            assert heading(3) == "# 3"
      """},
      ["m05_tag.py", "m05_doc.py"]),

    M("m06_repeat_param",
      "add a times parameter to repeat in m06_repeat and pass it from m06_banner",
      {"m06_repeat.py": """
        def repeat(s):
            return s * 2
      """,
       "m06_banner.py": """
        from m06_repeat import repeat

        def banner(s):
            return repeat(s)
      """},
      {"m06_repeat.py": """
        def repeat(s, times=1):
            return s * times
      """,
       "m06_banner.py": """
        from m06_repeat import repeat

        def banner(s):
            return repeat(s, 3)
      """},
      {"test_m06.py": """
        from m06_banner import banner

        def test_main():
            assert banner("ab") == "ababab"
      """},
      ["m06_repeat.py", "m06_banner.py"]),

    M("m07_rename_cycle",
      "fix the duplicate last element, rename cycle to concat_self in m07_cy, and update m07_ring",
      {"m07_cy.py": """
        def cycle(xs):
            return xs + xs[:-1]
      """,
       "m07_ring.py": """
        from m07_cy import cycle

        def doubled(xs):
            return cycle(xs)
      """},
      {"m07_cy.py": """
        def concat_self(xs):
            return xs + xs
      """,
       "m07_ring.py": """
        from m07_cy import concat_self

        def doubled(xs):
            return concat_self(xs)
      """},
      {"test_m07.py": """
        from m07_cy import concat_self
        from m07_ring import doubled

        def test_main():
            assert concat_self([1, 2]) == [1, 2, 1, 2]
            assert doubled([1, 2]) == [1, 2, 1, 2]
      """},
      ["m07_cy.py", "m07_ring.py"]),

    M("m08_join_sep",
      "add a separator parameter to glue in m08_join and pass a slash from m08_path",
      {"m08_join.py": """
        def glue(parts):
            return "-".join(parts)
      """,
       "m08_path.py": """
        from m08_join import glue

        def path(*parts):
            return glue(parts)
      """},
      {"m08_join.py": """
        def glue(parts, sep="-"):
            return sep.join(parts)
      """,
       "m08_path.py": """
        from m08_join import glue

        def path(*parts):
            return glue(parts, "/")
      """},
      {"test_m08.py": """
        from m08_path import path

        def test_main():
            assert path("a", "b") == "a/b"
      """},
      ["m08_join.py", "m08_path.py"]),
]


ALL_TASKS = TASKS + MULTI


def _blind_query(task: dict) -> str:
    """The task text with the answer's identifiers removed.

    A curated query names the function and file it is about, which is why
    retrieval looked trivial.  Dropping those names leaves a symptom
    description, so ranking has to work from meaning rather than a string
    the answer happens to contain.
    """
    names = set(re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", "\n".join(task["files"].values())))
    names.update(os.path.splitext(f)[0] for f in task["expected_files"])
    text = task["task"]
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(task: dict) -> list:
    src = "\n".join(task["files"].values()) + "\n" + task["task"]
    seen, out = set(), []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src):
        if len(word) >= 3 and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def distractors(task: dict, n: int = DISTRACTORS) -> dict:
    """Unrelated modules that share the task's vocabulary.

    They are never imported by the tests, so they cannot change pass/fail,
    but they give retrieval near-duplicate chunks with the same identifiers,
    strings, and words as the file that must be found, so lexical and dense
    ranking has to actually discriminate.
    """
    toks = _tokens(task)
    defs = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", "\n".join(task["files"].values()))
    base = defs[:3] or ["helper"]
    files = {}
    for i in range(n):
        start = (i * 3) % len(toks) if toks else 0
        note = " ".join(toks[start:][:6]) if toks else task["id"]
        lines = [f'"""Variant {i}: unrelated, but mentions {note}."""', ""]
        for j, name in enumerate(base):
            lines += [f"def {name}_alt{i}(*args, **kwargs):", f"    return {i} + {j}", ""]
        files[f"distractor_{task['id']}_{i}.py"] = "\n".join(lines) + "\n"
    return files


def build_suite() -> dict:
    return {"name": "curated", "tasks": ALL_TASKS}


def build_hard() -> dict:
    return {"name": "hard",
            "tasks": [{**task, "task": _blind_query(task),
                       "files": {**task["files"], **distractors(task)}}
                      for task in ALL_TASKS]}


def build_multifile() -> dict:
    return {"name": "multifile", "tasks": MULTI}


def build_all() -> list:
    return [(OUT, build_suite()), (HARD_OUT, build_hard()), (MULTI_OUT, build_multifile())]


def _run(root: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=root, capture_output=True, text=True, env=env)


def check() -> int:
    failures = []
    for task in ALL_TASKS:
        root = tempfile.mkdtemp(prefix=f"curated-{task['id']}-")
        try:
            for rel, content in task["files"].items():
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(content)
            buggy = _run(root)
            for test_id in task["must_keep_passing"]:
                node = test_id.split("::")[-1]
                if buggy.returncode != 0 and "FAILED" in buggy.stdout and node in buggy.stdout:
                    failures.append(f"{task['id']}: guard test fails on buggy code")
            main = task["expect_pass"][0]
            node, test_file = main.split("::")[-1], main.split("::")[0]
            if buggy.returncode == 0 or (node not in buggy.stdout and test_file not in buggy.stdout):
                failures.append(f"{task['id']}: main test does not fail on buggy code")
            for rel, content in task["fix"].items():
                with open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
                    fh.write(content)
            fixed = _run(root)
            if fixed.returncode != 0:
                failures.append(f"{task['id']}: tests do not pass after the fix\n{fixed.stdout[-800:]}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"ok: {len(ALL_TASKS)} tasks verified")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        code = check()
        if code:
            return code
    for path, suite in build_all():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(suite, fh, indent=2)
            fh.write("\n")
        print(f"wrote {path} ({len(suite['tasks'])} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
