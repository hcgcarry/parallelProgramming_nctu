#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <iostream>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define hybrid
#define buttom_up
#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define top_down
#define CTB_Param 14
#define CBT_Param 24
//#define VERBOSE
using namespace std;

inline void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

inline void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
omp_lock_t lck;
inline void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp for
    for (int i = 0; i < frontier->count; i++)
    {
        //cout << "thread num " << omp_get_num_threads() << endl;
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        int newDistanceForNeighBor = distances[node]+1;
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            int visited_marker=1;
            int* distancesOutGoingAddress = &distances[outgoing];
            //omp_set_lock(&lck);
            //#pragma omp critical
            /*
            {
                //#pragma omp atomic capture
                int curValue;
                int nextValue;
                do
                {
                    if(*distancesOutGoingAddress!=NOT_VISITED_MARKER){
                        visited_marker = 0;
                        break;
                    } 
                    curValue= NOT_VISITED_MARKER;
                    nextValue= newDistanceForNeighBor;
                } while (!__sync_bool_compare_and_swap(distancesOutGoingAddress, curValue, nextValue));
            }
            */
            {
                //#pragma omp atomic capture
                int curValue;
                int nextValue;
                do
                {
                    if(distances[outgoing]!=NOT_VISITED_MARKER){
                        visited_marker = 0;
                        break;
                    } 
                    curValue= NOT_VISITED_MARKER;
                    nextValue= newDistanceForNeighBor;
                } while (!__sync_bool_compare_and_swap(&distances[outgoing], curValue, nextValue));
            }
            /*
            {
                visited_marker = distances[outgoing];
                if (visited_marker == NOT_VISITED_MARKER)
                {
                    distances[outgoing] = distances[node] + 1;
                }
            }
            */
            //omp_unset_lock(&lck);
            if (visited_marker == 1)
            {
                int index;
                //#pragma omp critical(get_new_frontier_count)
                {
                    #pragma omp atomic capture
                    index = new_frontier->count++;
                }
                new_frontier->vertices[index] = outgoing;
            }
            /*
            if (visited_marker == NOT_VISITED_MARKER)
            {
                int curIndex;
                int nextIndex;
                do
                {
                    curIndex = new_frontier->count;
                    nextIndex = curIndex;
                    nextIndex++;
                } while (!__sync_bool_compare_and_swap(&(new_frontier->count), curIndex, nextIndex));
                new_frontier->vertices[curIndex] = outgoing;
            }
            /*
        */
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    #ifdef top_down
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    #pragma omp parallel
    {

        while (frontier->count != 0)
        {

#ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
#endif
            #pragma omp single

            {
                vertex_set_clear(new_frontier);
            }

            top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            printf("top down frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

            #pragma omp single
            {
                vertex_set *tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
            }
        }
    }
    #endif
}

typedef struct frontierBitMap
{
    bool *bitMap;
    int nodeCount;
    int maxNodeNum;
} FrontierBitMap;

inline void BitMapClear(FrontierBitMap *bitmap)
{
    bitmap->nodeCount = 0;

    #pragma omp for
    for (int i = 0; i < bitmap->maxNodeNum; i++)
    {
        bitmap->bitMap[i] = 0;
    }
}
inline void BitMapInit(FrontierBitMap *bitmap, int VertexNum)
{
    bitmap->maxNodeNum = VertexNum;
    bitmap->bitMap = (bool *)malloc(sizeof(int) * VertexNum);
    BitMapClear(bitmap);
}
inline void buttomUpStep(Graph graph, FrontierBitMap *frontier, FrontierBitMap *new_frontier, int *distances)
{
    //#pragma omp for schedule(dynamic,50)
    int  localNewFrontierCount=0;
    #pragma omp for 
    for (int i = 0; i < graph->num_nodes; i++)
    {
        //new_frontier->bitMap[i] =0;
        if (distances[i] == NOT_VISITED_MARKER)
        {
            //visit its neighbors
            int node = i;
            int start_edge = graph->incoming_starts[node];
            int end_edge = (node == graph->num_nodes - 1)
                               ? graph->num_edges
                               : graph->incoming_starts[node + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = graph->incoming_edges[neighbor];
                if (frontier->bitMap[incoming] == 1)
                {
                    //add cur node to new frontire
                    new_frontier->bitMap[node] = 1;
                    //#pragma omp atomic
		    localNewFrontierCount++;
		    //new_frontier->nodeCount=1;
                    distances[node] = distances[incoming] + 1;
                    break;
                }
            }
        }
    }
#pragma omp atomic 
    new_frontier->nodeCount+=localNewFrontierCount;
    //if(localNewFrontierCount>0) new_frontier->nodeCount=1;
	int curValue;
	int nextValue;
	/*
	do
	{
	    curValue= new_frontier->nodeCount
	    nextValue= newDistanceForNeighBor;
	} while (!__sync_bool_compare_and_swap(&new_frontier->nodeCount, curValue, nextValue));
    
	*/
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    #ifdef buttom_up
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    FrontierBitMap frontierObj;
    FrontierBitMap new_frontierObj;
    FrontierBitMap *frontier = &frontierObj;
    FrontierBitMap *new_frontier = &new_frontierObj;
    BitMapInit(frontier, graph->num_nodes);
    BitMapInit(new_frontier, graph->num_nodes);
    //#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    //frontier->nf = outgoing_size(graph, ROOT_NODE_ID);
    frontier->nodeCount = 1;
    frontier->bitMap[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;
    #pragma omp parallel
    {

        while (frontier->nodeCount != 0)
        {

#ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
#endif

            BitMapClear(new_frontier);


            buttomUpStep(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
            double end_time = CycleTimer::currentSeconds();
            printf("buttom up frontier=%-10d %.4f sec\n", frontier->nodeCount, end_time - start_time);
#endif
            // swap pointers
            #pragma omp single
            {
                FrontierBitMap *tmp = frontier;
                frontier = new_frontier;
                new_frontier = tmp;
            }
        }
    }
    #endif
}

void bfs_hybrid(Graph graph, solution *sol)
{
    #ifdef hybrid
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    //0 :topdown 1:buttomup
    int curIsTopDownOrButtomUp = 0;
    ////////////////////top down
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontierList = &list1;
    vertex_set *new_frontierList = &list2;
    // setup frontier with the root node
    frontierList->vertices[frontierList->count++] = ROOT_NODE_ID;

    //////////////// buttom up
    FrontierBitMap frontierObjBitMap;
    FrontierBitMap new_frontierObjBitMap;
    FrontierBitMap *frontierBitMap = &frontierObjBitMap;
    FrontierBitMap *new_frontierBitMap = &new_frontierObjBitMap;
    BitMapInit(frontierBitMap, graph->num_nodes);
    BitMapInit(new_frontierBitMap, graph->num_nodes);

    //////
    sol->distances[ROOT_NODE_ID] = 0;

    int mf = outgoing_size(graph, ROOT_NODE_ID);
    int nf;
    int mu;
    #pragma omp parallel
    {

        while ((curIsTopDownOrButtomUp == 0 && frontierList->count) || (curIsTopDownOrButtomUp == 1 && frontierBitMap->nodeCount))
        {
#ifdef VERBOSE
            double start_time = CycleTimer::currentSeconds();
#endif
            if (curIsTopDownOrButtomUp == 0)
            {
                #pragma omp single
                {
                    vertex_set_clear(new_frontierList);
                }
                //#pragma omp parallel
                {
                top_down_step(graph, frontierList, new_frontierList, sol->distances);
                }
#ifdef VERBOSE
                double end_time = CycleTimer::currentSeconds();
                printf("top down frontier=%-10d %.4f sec\n", frontierList->count, end_time - start_time);
#endif
                // swap pointers
                #pragma omp single
                {
                    vertex_set *tmp = frontierList;
                    frontierList = new_frontierList;
                    new_frontierList = tmp;
                // cacualte if need change to buttom up
                    mu = graph->num_edges;
                    mf = 0;
                }
                #pragma omp for reduction(-:mu)
                for (int i = 0; i < frontierList->count; i++)
                {
                    //#pragma omp atomic
                    //mf += outgoing_size(graph, frontierList->vertices[i]);
                    //j#pragma omp atomic
                    mu -= incoming_size(graph, frontierList->vertices[i]);
                }
                #pragma omp for reduction(+:mf)
                for (int i = 0; i < frontierList->count; i++)
                {
                    //#pragma omp atomic
                    mf += outgoing_size(graph, frontierList->vertices[i]);
                    //#pragma omp atomic
                    //mu -= incoming_size(graph, frontierList->vertices[i]);
                }
                int CTB = mu / CTB_Param;
                if (mf > CTB)
                {
                    curIsTopDownOrButtomUp = 1;
                    //#pragma omp parallel
                    {
                    BitMapClear(frontierBitMap);
                    }
                    #pragma omp for 
                    for (int i = 0; i < frontierList->count; i++)
                    {
                        frontierBitMap->bitMap[frontierList->vertices[i]] = 1;
                    }
                    frontierBitMap->nodeCount = frontierList->count;
                }
            }
            else
            {
#ifdef VERBOSE
                double start_time = CycleTimer::currentSeconds();
#endif
                //#pragma omp parallel
                {
                    BitMapClear(new_frontierBitMap);
                }

                //#pragma omp parallel
                {
                buttomUpStep(graph, frontierBitMap, new_frontierBitMap, sol->distances);
                }

#ifdef VERBOSE
                double end_time = CycleTimer::currentSeconds();
                printf("buttom up frontier=%-10d %.4f sec\n", frontierBitMap->nodeCount, end_time - start_time);
#endif
                // swap pointers
                #pragma omp single
                {
                    FrontierBitMap *tmp = frontierBitMap;
                    frontierBitMap = new_frontierBitMap;
                    new_frontierBitMap = tmp;
                }
                // caculate if need to switch to top down;
                nf = frontierBitMap->nodeCount;
                int CBT = graph->num_nodes / CBT_Param;
                if (nf < CBT)
                {
                    curIsTopDownOrButtomUp = 0;
                    vertex_set_clear(frontierList);
                    #pragma omp for
                    for (int i = 0; i < frontierBitMap->maxNodeNum; i++)
                    {
                        if (frontierBitMap->bitMap[i] == 1)
                        {
                            int index;
                            #pragma omp atomic capture
                            index = frontierList->count++;
                            frontierList->vertices[index]= i;
                        }
                    }
                }
            }
        }
    }
    #endif
}
