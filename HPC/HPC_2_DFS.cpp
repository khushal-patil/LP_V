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

    // Parallel DFS using stack
    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        vector<int> stack;

        stack.push_back(start);

        cout << "\nParallel DFS Traversal: ";

        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();

            if (visited[u]) continue;

            visited[u] = true;
            cout << u << " ";

            vector<int> to_push;

            #pragma omp parallel
            {
                vector<int> local_push;

                #pragma omp for nowait schedule(dynamic)
                for (int i = 0; i < adj[u].size(); i++) {
                    if (!visited[adj[u][i]]) {
                        local_push.push_back(adj[u][i]);
                    }
                }

                #pragma omp critical
                {
                    to_push.insert(to_push.end(),
                                   local_push.begin(),
                                   local_push.end());
                }
            }

            // push neighbors into stack
            for (int v : to_push) {
                stack.push_back(v);
            }
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
    cout << "Enter starting node for DFS: ";
    cin >> start;

    // validation
    if (start < 0 || start >= V) {
        cout << "Invalid start node!" << endl;
        return 0;
    }

    g.parallelDFS(start);

    return 0;
}




// Vertices: 5
// Edges: 4
// 0 1
// 0 2
// 1 3
// 2 4
// Start: 0

// Parallel DFS Traversal: 0 2 4 1 3

// “DFS goes deep first. From 0 it may go to 2 → 4, then backtrack to 1 → 3. Output may vary due to parallel execution.”