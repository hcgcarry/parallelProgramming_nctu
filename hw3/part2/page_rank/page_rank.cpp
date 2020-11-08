#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"
//#define debug

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / (double)numNodes;
  bool converged = false;
  double global_diff = 0;
  double *score_old = (double *)malloc(sizeof(double) * numNodes);
  for (int i = 0; i < numNodes; ++i)
  {
    score_old[i] = equal_prob;
  }

  while (!converged)
  {
    global_diff = 0.0;
    // compute score_new[vi] for all nodes vi:
    #ifdef debug
    printf("----- new iteration-----\n");
    printf("old score\n");
    for (int vi = 0; vi < num_nodes(g); vi++)
    {
      printf(" %d:%f", vi, score_old[vi]);
    }
    printf("\n");
    #endif
    for (int vi = 0; vi < num_nodes(g); vi++)
    {
      solution[vi] = 0.0;
      // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
      const Vertex *start = incoming_begin(g, vi);
      const Vertex *end = incoming_end(g, vi);
      for (const Vertex *vj = start; vj != end; vj++)
      {
        if (outgoing_size(g, *vj))
        {
          solution[vi] += (score_old[*vj] / (double)outgoing_size(g, *vj));
          #ifdef debug
          printf("-----------change in ------- \n ");
          printf("in vertex %d \n",*vj);
          printf("score old %f outgoing size %d\n", score_old[*vj] , outgoing_size(g, *vj));
          printf("solution[vi] += score_old[*vj] / outgoing_size(g, *vj)]\n");
          for (int vi = 0; vi < num_nodes(g); vi++)
          {
            printf(" %d:%f", vi, solution[vi]);
          }
          printf("-----------change in end------- \n ");
          #endif
        }
      }
      solution[vi] = (damping * solution[vi]) + ((1.0 - damping) /numNodes);
      //solution[vi] = (damping * solution[vi]) + (1.0 - damping) numNodes;
      #ifdef debug
      printf("-----------change middle ------- \n ");
      printf("solution[vi] = (damping * solution[vi]) + (1.0 - damping) / numNodes;\n");
      printf("damping*solution:%f,1-damping/numofnode%f ", (damping * solution[vi]) , (1.0 - damping) / numNodes);
      printf("middle solution\n");
      for (int vi = 0; vi < num_nodes(g); vi++)
      {
        printf(" %d:%f", vi, solution[vi]);
      }
      
      printf("\n-----------change middle end------- \n ");
      #endif
      for (const Vertex *vj = start; vj != end; vj++)
      {
        if (!outgoing_size(g, *vj))
        {
          solution[vi] += (damping * score_old[*vj]) / numNodes;
          #ifdef debug
          printf("-----------change below ------- \n ");
          printf("in vertex %d \n",*vj);
          printf("damping %f,score old %f, numNodes %d\n",damping, score_old[*vj] , numNodes);
          for (int vi = 0; vi < num_nodes(g); vi++)
          {
            printf(" %d:%f", vi, solution[vi]);
          }
          printf("-----------change below end------- \n ");
          #endif
        }
      }

      global_diff += abs(solution[vi] - score_old[vi]);
      #ifdef debug
      printf("global_diff %f , solution[vi] %f, score_old[vi] %f\n",global_diff,solution[vi],score_old[vi]);
      #endif
      score_old[vi] = solution[vi];
    }
    #ifdef debug
    printf("solution\n");
    for (int vi = 0; vi < num_nodes(g); vi++)
    {
      printf(" %d:%f", vi, solution[vi]);
    }
    printf("\n");
    #endif
    converged = (global_diff < convergence);
  }
  free(score_old);
  score_old = NULL;

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
