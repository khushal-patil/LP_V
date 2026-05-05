#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        vector<int> frontier;

        visited[start] = true;
        frontier.push_back(start);

        cout << "\nParallel BFS Traversal: ";

        while (!frontier.empty()) {

            for (int node : frontier)
                cout << node << " ";

            vector<int> next_frontier;

            #pragma omp parallel
            {
                vector<int> local_next;

                #pragma omp for nowait schedule(dynamic)
                for (int i = 0; i < frontier.size(); i++) {
                    int u = frontier[i];

                    for (int v : adj[u]) {
                        bool added = false;

                        #pragma omp critical
                        {
                            if (!visited[v]) {
                                visited[v] = true;
                                added = true;
                            }
                        }

                        if (added)
                            local_next.push_back(v);
                    }
                }

                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(),
                                         local_next.begin(),
                                         local_next.end());
                }
            }

            frontier = next_frontier;
        }

        cout << endl;
    }
};

int main() {
    int V, E;

    cout << "Enter number of vertices: ";
    cin >> V;

    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "Enter starting node for BFS: ";
    cin >> start;

    if (start < 0 || start >= V) {
        cout << "Invalid start node!" << endl;
        return 0;
    }

    g.parallelBFS(start);

    return 0;
}




// Enter number of vertices: 5
// Enter number of edges: 4
// Enter edges:
// 0 1
// 0 2
// 1 3
// 2 4
// Enter starting node: 0

// Parallel BFS Traversal: 0 1 2 3 4

// “Starting from node 0, BFS explores level by level. 
// First neighbors (1,2), then next level (3,4). Parallelism is applied while exploring neighbors.”