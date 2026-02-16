from queue import PriorityQueue

graph={
    'A': [('B',1), ('C',3)],
    'B': [('D',3),('E',1)],
    'C': [('F',5)],
    'D': [],
    'E': [('F',2)],
    'F': []
}


heuristic={'A':6,
           'B':4,
           'C':4,
           'D':6,
           'E':2,
           'F':0,
}

def a_star(start, goal):
  pq= PriorityQueue()
  pq.put((heuristic[start], [start],0))

  while not pq.empty():
    f, path, g = pq.get()
    current=path[-1]

    if current==goal:
      print("Path found:", ' -> '.join(path))
      print("Total cost:",g)
      return

    for neighbour, cost in graph.get(current, []):
      new_g=g + cost
      new_f=new_g + heuristic[neighbour]
      pq.put((new_f,path + [neighbour], new_g))

  print("No path found.")
a_star('A', 'F')
