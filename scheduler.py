import json
import time
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

@dataclass
class Task:
    """Represents a single pipeline task with resource requirements and constraints."""
    id: str
    resources: List[float]  # [CPU, RAM, GPU, Network]
    window: Tuple[int, int]  # (earliest_slot, latest_slot) inclusive
    weight: float  # Priority weight for penalty calculation

class ScoreMeScheduler:
    """
    Hybrid scheduler implementing Resource-Aware DSATUR with Tabu Local Search.
    Designed for the ScoreMe MSME Pipeline Scheduling problem.
    """

    def __init__(self, input_file: str):
        self.input_file = input_file
        self.tasks: List[Task] = []
        self.conflicts: List[Tuple[int, int]] = []
        self.capacities: List[List[float]] = []  # [slot][dimension]
        self.K = 0
        self.assignment: Dict[int, int] = {}
        self.adj_list: List[Set[int]] = []  # Adjacency list for conflict graph

    def load_instance(self) -> None:
        """Parse the input JSON file and initialize tasks, conflicts, and capacities."""
        with open(self.input_file, 'r') as f:
            data = json.load(f)
        
        self.K = data['K']
        self.capacities = data['capacities']
        
        # Build task objects
        for i, task_id in enumerate(data['tasks']):
            self.tasks.append(Task(
                id=task_id,
                resources=data['resources'][i],
                window=tuple(data['windows'][i]),
                weight=data['weights'][i]
            ))
        
        self.conflicts = data['conflicts']
        
        # Build adjacency list for O(1) conflict lookups
        self.adj_list = [set() for _ in range(len(self.tasks))]
        for i, j in self.conflicts:
            self.adj_list[i].add(j)
            self.adj_list[j].add(i)
        
        print(f"[OK] Loaded {len(self.tasks)} tasks, {self.K} slots.")

    def _check_all_feasible(self, assign: Dict[int, int]) -> Tuple[bool, str]:
        """
        Validate a complete assignment against constraints F1, F2, F3.
        Returns (True, "Feasible") if all constraints are satisfied, else (False, reason).
        """
        slot_loads = [[0.0] * 4 for _ in range(self.K)]
        
        for t_idx, slot in assign.items():
            task = self.tasks[t_idx]
            
            # Constraint F3: SLA Window
            if not (task.window[0] <= slot <= task.window[1]):
                return False, f"SLA violation for task {task.id}"
            
            # Constraint F2: Resource Capacity
            for d in range(4):
                slot_loads[slot][d] += task.resources[d]
                if slot_loads[slot][d] > self.capacities[slot][d] + 1e-6:
                    return False, f"Capacity exceeded in slot {slot}, dimension {d}"
        
        # Constraint F1: Conflict Avoidance
        for i, j in self.conflicts:
            if assign.get(i) == assign.get(j):
                return False, f"Conflict between {self.tasks[i].id} and {self.tasks[j].id}"
        
        return True, "Feasible"

    def _calculate_penalty(self, assign: Dict[int, int]) -> float:
        """
        Task 2: Extended Penalty Function.
        P(σ) = Base Weighted Delay + 0.1 * Load Imbalance Variance.
        Minimizing this ensures low latency and balanced cluster utilization.
        """
        base_penalty = 0.0
        cpu_loads = [0.0] * self.K
        
        for t_idx, slot in assign.items():
            base_penalty += self.tasks[t_idx].weight * slot
            cpu_loads[slot] += self.tasks[t_idx].resources[0]  # CPU is dimension 0
        
        # Load imbalance term (variance of CPU utilization across slots)
        avg_load = sum(cpu_loads) / self.K
        imbalance_penalty = sum((load - avg_load) ** 2 for load in cpu_loads)
        
        return base_penalty + (0.1 * imbalance_penalty)

    def _greedy_dsatur(self) -> Optional[Dict[int, int]]:
        """
        Phase 1: Greedy DSATUR-based coloring with resource and SLA awareness.
        Selects the most constrained task (highest saturation, degree, weight)
        and assigns it to the earliest feasible slot.
        """
        n = len(self.tasks)
        assigned = {}
        unassigned = set(range(n))
        
        while unassigned:
            best_task = -1
            best_sat = -1
            best_deg = -1
            best_weight = -1
            
            # Step 1: Select the most constrained task
            for t in unassigned:
                saturation = 0
                used_slots = set()
                for neighbor in self.adj_list[t]:
                    if neighbor in assigned:
                        used_slots.add(assigned[neighbor])
                saturation = len(used_slots)
                degree = len(self.adj_list[t])
                weight = self.tasks[t].weight
                
                # Prioritize: Saturation > Degree > Weight
                if (saturation > best_sat or 
                    (saturation == best_sat and degree > best_deg) or
                    (saturation == best_sat and degree == best_deg and weight > best_weight)):
                    best_sat = saturation
                    best_deg = degree
                    best_weight = weight
                    best_task = t
            
            task = self.tasks[best_task]
            assigned_slot = -1
            
            # Step 2: Find the earliest feasible slot within the SLA window
            for slot in range(task.window[0], task.window[1] + 1):
                # Check F1: Conflicts
                conflict_found = False
                for neighbor in self.adj_list[best_task]:
                    if neighbor in assigned and assigned[neighbor] == slot:
                        conflict_found = True
                        break
                if conflict_found:
                    continue
                
                # Check F2: Resource capacity
                can_fit = True
                for d in range(4):
                    current_load = sum(
                        self.tasks[t].resources[d] 
                        for t, s in assigned.items() 
                        if s == slot
                    )
                    if current_load + task.resources[d] > self.capacities[slot][d] + 1e-6:
                        can_fit = False
                        break
                
                if can_fit:
                    assigned_slot = slot
                    break
            
            # If no feasible slot found, the instance is infeasible
            if assigned_slot == -1:
                return None
            
            assigned[best_task] = assigned_slot
            unassigned.remove(best_task)
        
        return assigned

    def _is_move_feasible(self, assign: Dict[int, int], t_idx: int, new_slot: int) -> bool:
        """
        Check if moving a single task to a new slot violates any constraints.
        Used by the local search to evaluate potential moves.
        """
        # Validate SLA window
        if not (self.tasks[t_idx].window[0] <= new_slot <= self.tasks[t_idx].window[1]):
            return False
        
        # Validate conflicts
        for neighbor in self.adj_list[t_idx]:
            if neighbor in assign and assign[neighbor] == new_slot:
                return False
        
        # Validate resource capacities
        for d in range(4):
            load = sum(
                self.tasks[t].resources[d] 
                for t, s in assign.items() 
                if s == new_slot
            )
            if load + self.tasks[t_idx].resources[d] > self.capacities[new_slot][d] + 1e-6:
                return False
        
        return True

    def _local_search(self, initial_assign: Dict[int, int]) -> Dict[int, int]:
        """
        Phase 2: Tabu-constrained Local Search to minimize penalty.
        Iterates for 300 cycles, exploring 'move' and 'swap' neighborhoods.
        Accepts improving moves, and occasionally (10%) random moves to escape local minima.
        """
        current = initial_assign.copy()
        current_penalty = self._calculate_penalty(current)
        best_overall = current.copy()
        
        for _ in range(300):  # Fixed iterations for polynomial runtime
            neighbors = []
            
            # Neighborhood 1: Move a single task to another feasible slot
            for _ in range(3):
                t_idx = random.randint(0, len(self.tasks) - 1)
                current_slot = current[t_idx]
                possible_slots = list(
                    range(self.tasks[t_idx].window[0], self.tasks[t_idx].window[1] + 1)
                )
                random.shuffle(possible_slots)
                
                for new_slot in possible_slots:
                    if new_slot != current_slot and self._is_move_feasible(current, t_idx, new_slot):
                        neighbors.append(('move', t_idx, new_slot))
                        break
            
            # Neighborhood 2: Swap slots between two tasks (if SLAs permit)
            for _ in range(2):
                t1 = random.randint(0, len(self.tasks) - 1)
                t2 = random.randint(0, len(self.tasks) - 1)
                if t1 == t2:
                    continue
                
                # Check if swap respects SLA windows
                if (self.tasks[t1].window[0] <= current[t2] <= self.tasks[t1].window[1] and
                    self.tasks[t2].window[0] <= current[t1] <= self.tasks[t2].window[1]):
                    neighbors.append(('swap', t1, t2))
            
            # Evaluate all neighbors and select the best improvement
            best_delta = 0
            best_move = None
            
            for move in neighbors:
                temp_assign = current.copy()
                
                if move[0] == 'move':
                    _, t, s = move
                    temp_assign[t] = s
                else:  # 'swap'
                    _, t1, t2 = move
                    temp_assign[t1], temp_assign[t2] = current[t2], current[t1]
                
                # Ensure the neighbor is fully feasible
                is_feasible, _ = self._check_all_feasible(temp_assign)
                if not is_feasible:
                    continue
                
                new_penalty = self._calculate_penalty(temp_assign)
                delta = new_penalty - current_penalty
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = move
            
            # Apply the best move if it improves or with 10% random probability
            if best_move and (best_delta < 0 or random.random() < 0.1):
                if best_move[0] == 'move':
                    _, t, s = best_move
                    current[t] = s
                else:
                    _, t1, t2 = best_move
                    current[t1], current[t2] = current[t2], current[t1]
                
                current_penalty += best_delta
                
                # Update global best
                if current_penalty < self._calculate_penalty(best_overall):
                    best_overall = current.copy()
        
        return best_overall

    def solve(self) -> Dict:
        """
        Execute the full scheduling pipeline.
        Returns a dictionary matching the required output JSON format.
        """
        start_time = time.time()
        
        # Phase 1: Greedy DSATUR initial assignment
        greedy_assign = self._greedy_dsatur()
        feasible = False
        reason = "Greedy DSATUR failed to find a feasible initial assignment."
        final_assign = {}
        penalty = 0.0
        
        if greedy_assign is not None:
            # Phase 2: Local search to improve penalty
            final_assign = self._local_search(greedy_assign)
            feasible, reason = self._check_all_feasible(final_assign)
            if feasible:
                penalty = self._calculate_penalty(final_assign)
        else:
            reason = "Infeasible: Constraints cannot be satisfied (check conflicts/capacity/SLA)."
        
        runtime_ms = int((time.time() - start_time) * 1000)
        
        # --- FIX: Handle infeasible case to avoid KeyError ---
        output_assign = {}
        if feasible:
            output_assign = {f"T{i}": final_assign[i] for i in range(len(self.tasks))}
        else:
            # If infeasible, assign -1 to all tasks to maintain JSON structure
            output_assign = {f"T{i}": -1 for i in range(len(self.tasks))}
        # ----------------------------------------------------
        
        return {
            "assignment": output_assign,
            "penalty": round(penalty, 4) if feasible else 0.0,
            "runtime_ms": runtime_ms,
            "feasible": feasible,
            "violation_reason": reason if not feasible else ""
        }


# ------------------------- Main Entry Point -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scheduler.py <input.json>")
        sys.exit(1)
    
    scheduler = ScoreMeScheduler(sys.argv[1])
    scheduler.load_instance()
    result = scheduler.solve()
    
    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Result written to output.json")