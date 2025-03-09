class Chromosome:
    def __init__(self, task_parts=None, mutation_parts=None):
        self.task_parts = task_parts or []
        self.mutation_parts = mutation_parts or []
        
    def combine(self, other):
        """Combine two chromosomes by merging their parts"""
        new_task = self._merge_parts(self.task_parts, other.task_parts)
        new_mutation = self._merge_parts(self.mutation_parts, other.mutation_parts)
        return Chromosome(new_task, new_mutation)
        
    def _merge_parts(self, parts1, parts2):
        """Merge two sets of parts using crossover"""
        if not parts1 or not parts2:
            return parts1 or parts2
            
        min_len = min(len(parts1), len(parts2))
        crossover_point = random.randint(0, min_len)
        return parts1[:crossover_point] + parts2[crossover_point:]
        
    def mutate(self, mutation_rate=0.5):
        """Apply mutations to the chromosome"""
        if random.random() < mutation_rate:
            self.mutation_parts = self._shuffle_parts(self.mutation_parts)
            
    def _shuffle_parts(self, parts):
        """Randomly shuffle parts while maintaining structure"""
        if not parts:
            return parts
            
        # Split into instruction blocks
        blocks = []
        current_block = []
        for part in parts:
            if part.endswith('.'):
                current_block.append(part)
                blocks.append(current_block)
                current_block = []
            else:
                current_block.append(part)
                
        # Shuffle blocks
        random.shuffle(blocks)
        
        # Flatten back into parts
        return [part for block in blocks for part in block]
        
    def to_prompt(self):
        """Convert chromosome to executable prompt"""
        task = ' '.join(self.task_parts)
        mutation = ' '.join(self.mutation_parts)
        return f"{task} {mutation}".strip()
