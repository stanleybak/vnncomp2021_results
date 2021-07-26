"""
Process vnncomp results

Stanley Bak
"""

import glob
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

class ToolResult:
    """Tool's result"""

    # columns
    CATEGORY = 0
    NETWORK = 1
    PROP = 2
    PREPARE_TIME = 3
    RESULT = 4
    RUN_TIME = 5

    all_categories = set()

    # stats
    num_verified = defaultdict(int) # number of benchmarks verified
    num_violated = defaultdict(int) 
    num_holds = defaultdict(int)
    incorrect_results = defaultdict(int)

    num_categories = defaultdict(int)

    def __init__(self, tool_name, csv_path, cpu_benchmarks, skip_benchmarks):
        assert "csv" in csv_path

        self.tool_name = tool_name
        self.category_to_list = defaultdict(list) # maps category -> list of results

        self.skip_benchmarks = skip_benchmarks
        self.cpu_benchmarks = cpu_benchmarks
        self.gpu_overhead = np.inf # default overhead
        self.cpu_overhead = np.inf # if using separate overhead for cpu
        
        self.max_prepare = 0.0

        self.load(csv_path)

    def result_instance_str(self, cat, index):
        """get a string representation of the instance for the given category and index"""

        row = self.category_to_list[cat][index]

        net = row[ToolResult.NETWORK]
        prop = row[ToolResult.PROP]

        return Path(net).stem + "-" + Path(prop).stem

    def single_result(self, cat, index):
        """get result_str, runtime of tool, after subtracting overhead"""

        row = self.category_to_list[cat][index]

        res = row[ToolResult.RESULT]
        t = float(row[ToolResult.RUN_TIME])

        t -= self.cpu_overhead if cat in self.cpu_benchmarks else self.gpu_overhead

        # all results less than 1.0 second are treated the same
        if t < 1.0:
            t = 1.0

        return res, t

    def load(self, csv_path):
        """load data from file"""

        unexpected_results = set()
                
        with open(csv_path, newline='') as csvfile:
            for row in csv.reader(csvfile):
                # rename results
                row[ToolResult.RESULT] = row[ToolResult.RESULT].lower()

                substitutions = [('unsat', 'holds'),
                                 ('sat', 'violated'),
                                 ('no_result_in_file', 'unknown'),
                                 ('prepare_instance_error_', 'unknown'),
                                 ('run_instance_timeout', 'timeout'),
                                 ('error_exit_code_', 'error'),
                                 ]

                for from_prefix, to_str in substitutions:
                    if row[ToolResult.RESULT] == '': # don't use '' as prefix
                        row[ToolResult.RESULT] = 'unknown'
                    elif row[ToolResult.RESULT].startswith(from_prefix):
                        row[ToolResult.RESULT] = to_str

                network = row[ToolResult.NETWORK]
                result = row[ToolResult.RESULT]
                cat = row[ToolResult.CATEGORY]
                prepare_time = float(row[ToolResult.PREPARE_TIME])
                run_time = float(row[ToolResult.RUN_TIME])

                if cat in self.skip_benchmarks:
                    result = row[ToolResult.RESULT] = "unknown"

                if not ("test_nano" in network or "test_tiny" in network):
                    self.category_to_list[cat].append(row)

                if result not in ["holds", "violated", "timeout", "error", "unknown"]:
                    unexpected_results.add(result)

                if result in ["holds", "violated"]:
                    if cat in self.cpu_benchmarks:
                        self.cpu_overhead = min(self.cpu_overhead, run_time)
                    else:
                        self.gpu_overhead = min(self.gpu_overhead, run_time)
                        
                    self.max_prepare = max(self.max_prepare, prepare_time)

        assert not unexpected_results, f"Unexpected results: {unexpected_results}"

        print(f"Loaded {self.tool_name}, default-overhead (gpu): {round(self.gpu_overhead, 1)}s," + \
              f"cpu-overhead: {round(self.cpu_overhead, 1)}s, " + \
              f"prepare time: {round(self.max_prepare, 1)}s")

        self.delete_empty_categories()

    def delete_empty_categories(self):
        """delete categories without successful measurements"""

        to_remove = ["test"]

        for key in self.category_to_list.keys():
            rows = self.category_to_list[key]

            should_remove = True

            for row in rows:
                result = row[ToolResult.RESULT]

                if result in ('holds', 'violated'):
                    
                    should_remove = False
                    break

            if should_remove:
                to_remove.append(key)
            elif key != "test":
                ToolResult.all_categories.add(key)

        for key in to_remove:
            print(f"deleting {key} in tool {self.tool_name}")
            del self.category_to_list[key]

        ToolResult.num_categories[self.tool_name] = len(self.category_to_list)

def compare_results(result_list, resolve_conflicts):
    """compare results across tools"""

    min_percent = 0 # minimum percent for total score

    total_score = defaultdict(int)
    all_cats = {}

    for cat in ToolResult.all_categories:
        print(f"\nCategory {cat}:")

        cat_score = defaultdict(int)
        all_cats[cat] = cat_score

        num_rows = 0

        participating_tools = []

        for tool_result in result_list:
            cat_dict = tool_result.category_to_list

            if not cat in cat_dict:
                continue
            
            rows = cat_dict[cat]
            assert num_rows == 0 or len(rows) == num_rows, f"tool {tool_result.tool_name}, cat {cat}, " + \
                f"got {len(rows)} rows expected {num_rows}"

            if num_rows == 0:
                num_rows = len(rows)
                print(f"Category {cat} has {num_rows} (from {tool_result.tool_name})")

            participating_tools.append(tool_result)

        # work with participating tools only
        tool_names = [t.tool_name for t in participating_tools]
        print(f"{len(participating_tools)} participating tools: {tool_names}")
        table_rows = []

        for index in range(num_rows):
            rand_gen_succeeded = False
            times_holds = []
            times_violated = []
            
            table_row = []
            table_rows.append(table_row)
            instance_str = participating_tools[0].result_instance_str(cat, index)
            table_row.append(instance_str)

            for t in participating_tools:
                res, secs = t.single_result(cat, index)

                if res == "unknown":
                    table_row.append("-")
                    continue

                if not res in ["holds", "violated"]:
                    table_row.append(res)
                    continue

                if res == "holds":
                    times_holds.append(secs)
                else:
                    assert res == "violated"
                    times_violated.append(secs)

                table_row.append(f"{round(secs, 1)} ({res[0]})")

                if t.tool_name == "randgen":
                    assert res == "violated"
                    rand_gen_succeeded = True

            print()

            if times_holds and times_violated:
                print(f"WARNING: multiple results for index {index}. Violated: {len(times_violated)}, " + \
                      f"Holds: {len(times_holds)}")
                table_row.append('*multiple results*')
                
            print(f"Row: {table_row}")
            
            for t in participating_tools:
                res, secs = t.single_result(cat, index)
                
                score = get_score(t.tool_name, res, secs, rand_gen_succeeded, times_holds, times_violated,
                                  resolve_conflicts=resolve_conflicts)
                print(f"{index}: {t.tool_name} score: {score}")

                cat_score[t.tool_name] += score

        print(f"--------------------")
        print(", ".join(tool_names))

        for table_row in table_rows:
              print(", ".join(table_row))

        print(f"---------\nCategory {cat}:")

        max_score = max(cat_score.values())
                
        for tool, score in cat_score.items():
            percent = max(min_percent, 100 * score / max_score)
            print(f"{tool}: {score} ({round(percent, 2)}%)")

            if cat != 'cifar2020':
                total_score[tool] += percent

    print("\n###############")
    print("### Summary ###")
    print("###############")

    for cat in sorted(all_cats.keys()):
        cat_score = all_cats[cat]
        
        print(f"\nCategory {cat} (conflicts={resolve_conflicts}):")
        res_list = []
        max_score = max(cat_score.values())
                
        for tool, score in cat_score.items():
            percent = max(min_percent, 100 * score / max_score)
            #desc = f"{tool}: {score} ({round(percent, 2)}%)"
            desc = f"{tool},{score},{round(percent, 2)}%"

            res_list.append((percent, desc))

        for i, s in enumerate(reversed(sorted(res_list))):
            #print(f"{i+1}. {s[1]}")
            print(f"{s[1]}")

    res_list = []

    print(f"\nTotal Score (conflicts={resolve_conflicts}):")
    for tool, score in total_score.items():
        desc = f"{tool}: {round(score, 2)}"

        res_list.append((score, desc))

    for i, s in enumerate(reversed(sorted(res_list))):
        print(f"{i+1}. {s[1]}")

def get_score(tool_name, res, secs, rand_gen_succeded, times_holds, times_violated, resolve_conflicts):
    """Get the score for the given result

    Correct hold: 10 points
    Correct violated (where random tests did not succeed): 10 points
    Correct violated (where random test succeeded): 1 point
    Incorrect result: -100 points

    Time bonus: 
        The fastest tool for each solved instance will receive +2 points. 
        The second fastest tool will receive +1 point.
        If two tools have runtimes within 0.2 seconds, we will consider them the same runtime.
    """

    # how to resolve conflicts (some tools output holds others output violated)
    # "voting": majority rules, tie = ighore
    # "odd_one_out": only if single tool has mismatch, assume it's wrong
    # "ignore": ignore all conflicts
    assert resolve_conflicts in ["voting", "odd_one_out", "ignore"]

    num_holds = len(times_holds)
    num_violated = len(times_violated)

    if not res in ["holds", "violated"] or num_holds == num_violated:
        score = 0
    elif resolve_conflicts == "ignore" and num_holds > 0 and num_violated > 0:
        score = 0
    elif resolve_conflicts == "odd_one_out" and num_holds > 1 and num_violated > 1:
        score = 0
    elif rand_gen_succeded:
        assert res == "violated"
        score = 1

        ToolResult.num_verified[tool_name] += 1
        ToolResult.num_violated[tool_name] += 1
    elif num_holds > num_violated and res == "violated":
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
    elif num_violated > num_holds and res == "holds":
        score = -100
        ToolResult.incorrect_results[tool_name] += 1
    else:
        # correct result!

        ToolResult.num_verified[tool_name] += 1

        if res == "holds":
            times = times_holds.copy()
            ToolResult.num_holds[tool_name] += 1
        else:
            assert res == "violated"
            times = times_violated.copy()
            ToolResult.num_violated[tool_name] += 1
        
        score = 10

        min_time = min(times)

        if secs < min_time + 0.2:
            score += 2
        else:
            times.remove(min_time)
            second_time = min(times)

            if secs < second_time + 0.2:
                score += 1

    return score

def print_stats(result_list):
    """print stats about measurements"""

    print('\n------- Stats ----------')

    print("\nOverhead:")
    olist = []

    for r in result_list:
        olist.append((r.gpu_overhead, r.cpu_overhead, r.tool_name))
        
    for i, n in enumerate(sorted(olist)):
        print(f"{i+1}. {n}")

    items = [("Num Categories Completed", ToolResult.num_categories),
             ("Num Verified", ToolResult.num_verified),
             ("Num Violated", ToolResult.num_violated),
             ("Num Holds", ToolResult.num_holds),
             ('Judged as "Incorrect" Results', ToolResult.incorrect_results),
             ]

    for label, d in items:
        print(f"\n{label}:")

        l = []

        for name, count in d.items():
            l.append((count, name))
        
        for i, s in enumerate(reversed(sorted(l))):
            print(f"{i+1},{s[1]},{s[0]}")

def main():
    """main entry point"""

    # use single overhead for all tools. False will have two different overheads for ERAN depending on CPU/GPU
    single_overhead = True
    print(f"using single_overhead={single_overhead}")

    # how to resolve conflicts (some tools output holds others output violated)
    # "voting": majority rules, tie = ighore
    # "odd_one_out": only if single tool has mismatch, assume it's wrong
    # "ignore": ignore all conflicts
    # resolve_conflicts = "odd_one_out"
    resolve_conflicts = "odd_one_out"
    print(f"using resolve_conflicts={resolve_conflicts}")

    #####################################3
    csv_list = glob.glob("results_csv/*.csv")
    tool_list = [Path(c).stem for c in csv_list]
    result_list = []

    cpu_benchmarks = {x: [] for x in tool_list}
    skip_benchmarks = {x: [] for x in tool_list}
    skip_benchmarks['RPM'] = ['mnistfc']
    
    if not single_overhead: # Define a dict with the cpu_only benchmarks for each tool
        cpu_benchmarks["ERAN"] = ["acasxu", "eran"]

        # doesn't make much difference:
        #for t in tool_list:
        #    cpu_benchmarks[t] = ["acasxu", "eran"]

    for csv_path, tool_name in zip(csv_list, tool_list):
        tr = ToolResult(tool_name, csv_path, cpu_benchmarks[tool_name], skip_benchmarks[tool_name])
        result_list.append(tr)

    # compare results across tools
    compare_results(result_list, resolve_conflicts)

    print_stats(result_list)

if __name__ == "__main__":
    main()
