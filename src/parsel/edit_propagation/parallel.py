import multiprocessing as mp

import numpy as np

from ..edit_dsl.edits import HIGHER_TYPE_HINTS, MAIN_VAR, EditGen, KeepFixed
from ..shape_system.relations import FeatureRelation, PrimitiveRelation, Relation
from .proposer import (
    MIN_EDIT_CANDIDATES,
    EditProposer,
    get_additional_edit_candidates,
    get_edits_to_try,
    relation_based_edit_candidates,
    solve_edits,
    solve_edits_breaking,
    type_based_prune,
)
from .scoring import (
    merge_finalized_edits,
    remove_high_distortion,
    remove_null_edits,
    select_new_edits,
    sort_by_typed_distortion_energy,
)

MIN_FOR_PARALLEL = 5
MIN_FOR_PARALLEL_BREAKING = 5
MAX_TO_TRY = 7
MAX_KEEP_FIXED_TO_TRY = 2
class ParallelEditProposer(EditProposer):
    
    def __init__(self, heuristic="intrinsic"):
        self.broken_relations = []
        self.n_procs = 16
        self.heuristic  = heuristic
                            
    def solve_edit_by_type(self, operand, edit_type, shape, try_extensions=True, reject_null=False):
        if edit_type is None:
            edit_types = HIGHER_TYPE_HINTS[edit_type]
        elif isinstance(edit_type, str):
            edit_types = HIGHER_TYPE_HINTS[edit_type]
        else:
            edit_types = edit_type
        
        if isinstance(operand, Relation):
            selected_candidates = operand.gather_update_options()
            selected_candidates = type_based_prune(operand, selected_candidates, edit_types)
            if not selected_candidates:
                return None, None
            return selected_candidates[0].employ(operand), None
        else:
            retained_relations = self.get_relevant_relations(operand)
            retained_relations = [x for x in retained_relations if isinstance(x, FeatureRelation)]
        edits_to_try = gather_base_edits(operand, retained_relations)
        edits_to_try = type_based_prune(operand, edits_to_try, edit_types)

        selected_candidates = self.parallel_edits_wrapper(operand, retained_relations, edits_to_try, reject_null=reject_null)
        if len(selected_candidates) < MIN_EDIT_CANDIDATES:
            new_edits_to_try = gather_additional_edit_variants(operand, retained_relations, edit_types, shape, edits_to_try)
            new_edits_to_try = type_based_prune(operand, new_edits_to_try, edit_types)
            selected_candidates = self.parallel_edits_wrapper(operand, retained_relations, new_edits_to_try, reject_null=reject_null)
        if len(selected_candidates) > 0:
            solved_edit = selected_candidates[0].employ(operand)
            least_breaking = None
        else:
            # Need to solve and record.
            # solved_edit = None
            # least_breaking = None
            # return solved_edit, least_breaking
            # TODO: are just the basic edits enough?
            # Maybe also derive from siblings?
            # new_edits_to_try = gather_additional_edit_variants(operand, retained_relations, edit_types, shape, edits_to_try)
            edits_to_try.extend(new_edits_to_try)

            solutions = self.parallel_breaking_wrapper(operand, retained_relations, edits_to_try)
            solutions = [x for x in solutions if MAIN_VAR in x[0].amount.free_symbols]
            
            selected_candidates = [x[0] for x in solutions]
            # first create the valid set:
            selected_candidates, subset_solutions = merge_finalized_edits(operand, selected_candidates, solutions)
            pruned_selected_candidates, pruned_subset_solutions = remove_high_distortion(operand, selected_candidates, subset_solutions)
            if len(pruned_selected_candidates) > 0:
                selected_candidates = pruned_selected_candidates
                subset_solutions = pruned_subset_solutions

            broken_relations = [x[1] for x in subset_solutions]
            broken_equations = [x[2] for x in subset_solutions]
            if len(broken_relations) == 0:
                solved_edit = None
                least_breaking = None
                return solved_edit, least_breaking
            
            # min_cost = np.min([len(x) for x in broken_relations])
            min_cost = np.min([len(x) for x in broken_equations])
            # Just select least breaking?
            selected_candidates = [x[0] for x in subset_solutions if len(x[2]) == min_cost]
            broken_relations = [x[1] for x in subset_solutions if len(x[2]) == min_cost]
            # join the lists in broken_relations
            # selected_candidates, broken_relations = remove_null_edits(selected_candidates, broken_relations)
            pre_geom_solutions = len(selected_candidates)
            selected_candidates, broken_relations = sort_by_typed_distortion_energy(operand, selected_candidates, broken_relations, heuristic=self.heuristic)
            
            # all_broken_relations = list(set(retained_relations))
            # final_list = []
            # for ind, candidate in enumerate(selected_candidates):
            #     cur_broken_rel = broken_relations[ind]
            #     kept_relations = [x for x in all_broken_relations if x not in cur_broken_rel]
            #     cur_value = np.max([RELATION_VALUE[x.__class__] for x in kept_relations])
            #     final_list.append((candidate, cur_broken_rel, cur_value))
            # final_list = sorted(final_list, reverse=True, key=lambda x: x[2])
            # selected_candidates = [x[0] for x in final_list]
            # broken_relations = [x[1] for x in final_list]

            if len(selected_candidates) > 0:
                solved_edit = None
                least_breaking = (selected_candidates[0].employ(operand), broken_relations[0])
            else:
                # if len(selected_candidates)  > 0:
                solved_edit = None
                least_breaking = None
        return solved_edit, least_breaking
    
    def parallel_breaking_wrapper(self, operand, retained_relations, edits_to_try):
        if len(edits_to_try) < MIN_FOR_PARALLEL_BREAKING:
            output = solve_edits_breaking(operand, retained_relations, edits_to_try)
            return output
        else:
            n_procs = min(self.n_procs, np.round(len(edits_to_try)/MIN_FOR_PARALLEL_BREAKING).astype(int))
            output = self.parallel_breaking(operand, retained_relations, edits_to_try, n_procs=n_procs)
            return output
    
    def parallel_breaking(self, operand, retained_relations, edits_to_try, n_procs=None):
        if n_procs is None:
            n_procs = self.n_procs

        per_process_edits = [[] for x in range(n_procs)]
        for ind, edit in enumerate(edits_to_try):
            process_ind = ind % n_procs
            per_process_edits[process_ind].append(edit)
            
        queue = mp.Manager().Queue()
        single_solution_mode, hasten_search, solver_mode = True, True, "BASE"
        processes = []
        for proc_id in range(n_procs):
            
            p = mp.Process(target=parallel_breaking_solve_call, args=(operand, 
                                                             retained_relations, 
                                                             per_process_edits[proc_id],
                                                             queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        outputs = []
        while not queue.empty():
            outputs.extend(queue.get())
        return outputs
    
    def parallel_edits_wrapper(self, operand, retained_relations, edits_to_try, reject_null=False):

        if len(edits_to_try) < MIN_FOR_PARALLEL:
            selected_candidates = solve_edits(operand, retained_relations, edits_to_try, single_solution_mode=True, hasten_search=True, solver_mode="BASE")

            if reject_null:
                selected_candidates = remove_null_edits(selected_candidates)
            selected_candidates = merge_finalized_edits(operand, selected_candidates)
            # selected_candidates = remove_flips(operand, selected_candidates)
            selected_candidates = remove_high_distortion(operand, selected_candidates, return_sorted=True)
            selected_candidates = sort_by_typed_distortion_energy(operand, selected_candidates, heuristic=self.heuristic)
            return selected_candidates
        else:
            n_procs = min(self.n_procs, np.round(len(edits_to_try)/MIN_FOR_PARALLEL).astype(int))
            output = self.parallel_edits(operand, retained_relations, edits_to_try, reject_null=reject_null, n_procs=n_procs)
            return output


    def parallel_edits(self, operand, retained_relations, edits_to_try, reject_null=False, n_procs=None):
        if n_procs is None:
            n_procs = self.n_procs
        per_process_edits = [[] for x in range(n_procs)]
        for ind, edit in enumerate(edits_to_try):
            process_ind = ind % n_procs
            per_process_edits[process_ind].append(edit)
            
        queue = mp.Manager().Queue()
        single_solution_mode, hasten_search, solver_mode = True, True, "BASE"
        processes = []
        for proc_id in range(n_procs):
            
            p = mp.Process(target=parallel_solve_call, args=(operand, 
                                                             retained_relations, 
                                                             per_process_edits[proc_id],
                   single_solution_mode, hasten_search, solver_mode, queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        outputs = []
        while not queue.empty():
            outputs.extend(queue.get())
        all_edit_candidates = outputs
        
        if reject_null:
            selected_candidates = remove_null_edits(all_edit_candidates)
        else:
            selected_candidates = all_edit_candidates

        selected_candidates = merge_finalized_edits(operand, selected_candidates)
        selected_candidates = remove_high_distortion(operand, selected_candidates)
        selected_candidates = sort_by_typed_distortion_energy(operand, selected_candidates, heuristic=self.heuristic)
        # selected_candidates = remove_geometric(operand, selected_candidates, heuristic=self.heuristic)
        return selected_candidates
    
def gather_base_edits(operand, retained_relations):
    sibling_edits_to_try = []
    related_parts = []
    for relation in retained_relations:
        if isinstance(relation, FeatureRelation):
            parts = [x.primitive.part for x in relation.features]
        elif isinstance(relation, PrimitiveRelation):
            parts = [x.part for x in relation.primitives]
        other_parts = [x for x in parts if x != operand]
        related_parts.extend(other_parts)
    related_parts = list(set(related_parts))
    for other_part in related_parts:
        sibling_edits = other_part.primitive.edit_sequence
        sibling_edits_to_try.extend(sibling_edits)
    sibling_edits_to_try = [x for x in sibling_edits_to_try if not isinstance(x, KeepFixed)]
    sibling_edit_gens = []
    for edit in sibling_edits_to_try:
        edit_gen = EditGen(edit.__class__, edit.params)
        sibling_edit_gens.append(edit_gen)
    edits_to_try = get_edits_to_try(operand)
    edits_to_try.extend(sibling_edit_gens)
    return edits_to_try
    
    
def gather_additional_edit_variants(operand, retained_relations, type_hints, shape, edits_to_try):
    
    # Generate edits from existing validated relations
    new_edits_to_try = relation_based_edit_candidates(operand, shape, edits_to_try)
    # try objects connected to it
    sibling_edits_to_try = []
    related_parts = []
    for relation in retained_relations:
        if isinstance(relation, FeatureRelation):
            parts = [x.primitive.part for x in relation.features]
        elif isinstance(relation, PrimitiveRelation):
            parts = [x.part for x in relation.primitives]
        other_parts = [x for x in parts if x != operand]
        related_parts.extend(other_parts)
    related_parts = list(set(related_parts))
    for other_part in related_parts:
        if other_part.full_label == "ground":
            continue
        sibling_edits = other_part.primitive.edit_sequence
        sibling_edits_to_try.extend(sibling_edits)
    # sibling_edits_to_try = [x for x in sibling_edits_to_try if not isinstance(x, KeepFixed)]
    fixed_siblings = [x for x in sibling_edits_to_try if isinstance(x, KeepFixed)]
    not_fixed = [x for x in sibling_edits_to_try if not isinstance(x, KeepFixed)]
    reminder_count = max(0, MAX_TO_TRY - len(not_fixed))
    sibling_edits = not_fixed + fixed_siblings[:MAX_KEEP_FIXED_TO_TRY]
    for edit_ind, edit in enumerate(sibling_edits):
        # use other under edits. 
        newer_edits_to_try = get_additional_edit_candidates(operand, edit)
        new_edits_to_try.extend(newer_edits_to_try)
    new_edits_to_try = select_new_edits(operand, edits_to_try, new_edits_to_try)
    return new_edits_to_try

def parallel_solve_call(target_part, retained_relations, edits_to_try,
                   single_solution_mode=True,
                   hasten_search=True,
                   solver_mode="BASE",
                   queue=None):
    
    candidates = solve_edits(target_part, retained_relations, edits_to_try,
                   single_solution_mode, hasten_search, solver_mode)
    queue.put(candidates)
    

def parallel_breaking_solve_call(operand, retained_relations, edits_to_try, queue):
    candidates = solve_edits_breaking(operand, retained_relations, edits_to_try)
    queue.put(candidates)
    
    
