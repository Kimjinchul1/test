#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

// Define a structure for a neuron
struct Neuron {
    double* weights;
    double bias;
    int num_inputs;
};

// Define a structure for a layer of neurons
struct Layer {
    struct Neuron* neurons;
    int num_neurons;
};

// Define a structure for a graph node (for BFS and DFS)
struct Node {
    int value;
    struct Node** neighbors;
    int num_neighbors;
    bool visited;
};

// Function to create a new neuron with dynamic memory allocation
struct Neuron* create_neuron(int num_inputs) {
    struct Neuron* neuron = (struct Neuron*)malloc(sizeof(struct Neuron));
    neuron->num_inputs = num_inputs;
    neuron->weights = (double*)malloc(num_inputs * sizeof(double));
    neuron->bias = ((double)rand() / RAND_MAX) * 2 - 1; // Random bias between -1 and 1

    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random weights between -1 and 1
    }

    return neuron;
}

// Function to create a new layer with dynamic memory allocation
struct Layer* create_layer(int num_neurons, int num_inputs) {
    struct Layer* layer = (struct Layer*)malloc(sizeof(struct Layer));
    layer->num_neurons = num_neurons;
    layer->neurons = (struct Neuron*)malloc(num_neurons * sizeof(struct Neuron));

    for (int i = 0; i < num_neurons; i++) {
        layer->neurons[i] = *create_neuron(num_inputs);
    }

    return layer;
}

// Activation function (ReLU)
double relu(double x) {
    return x > 0 ? x : 0;
}

// Forward pass for a single neuron
double neuron_forward(struct Neuron* neuron, double* inputs) {
    double sum = neuron->bias;
    for (int i = 0; i < neuron->num_inputs; i++) {
        sum += neuron->weights[i] * inputs[i];
    }
    return relu(sum);
}

// Forward pass for a layer
void layer_forward(struct Layer* layer, double* inputs, double* outputs) {
    for (int i = 0; i < layer->num_neurons; i++) {
        outputs[i] = neuron_forward(&layer->neurons[i], inputs);
    }
}

// Free memory allocated for a neuron
void free_neuron(struct Neuron* neuron) {
    free(neuron->weights);
    free(neuron);
}

// Free memory allocated for a layer
void free_layer(struct Layer* layer) {
    for (int i = 0; i < layer->num_neurons; i++) {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
    free(layer);
}

// Create a graph node
struct Node* create_node(int value) {
    struct Node* node = (struct Node*)malloc(sizeof(struct Node));
    node->value = value;
    node->neighbors = NULL;
    node->num_neighbors = 0;
    node->visited = false;
    return node;
}

// Add an edge to the graph
void add_edge(struct Node* node1, struct Node* node2) {
    node1->num_neighbors++;
    node2->num_neighbors++;
    node1->neighbors = (struct Node**)realloc(node1->neighbors, node1->num_neighbors * sizeof(struct Node*));
    node2->neighbors = (struct Node**)realloc(node2->neighbors, node2->num_neighbors * sizeof(struct Node*));
    node1->neighbors[node1->num_neighbors - 1] = node2;
    node2->neighbors[node2->num_neighbors - 1] = node1;
}

// Queue structure for BFS
struct Queue {
    struct Node** array;
    int front;
    int rear;
    int capacity;
};

// Create a queue
struct Queue* create_queue(int capacity) {
    struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
    queue->capacity = capacity;
    queue->front = queue->rear = -1;
    queue->array = (struct Node**)malloc(queue->capacity * sizeof(struct Node*));
    return queue;
}

// Check if queue is full
int is_full(struct Queue* queue) {
    return ((queue->rear + 1) % queue->capacity == queue->front);
}

// Check if queue is empty
int is_empty(struct Queue* queue) {
    return (queue->front == -1);
}

// Add an item to the queue
void enqueue(struct Queue* queue, struct Node* item) {
    if (is_full(queue)) return;
    queue->rear = (queue->rear + 1) % queue->capacity;
    queue->array[queue->rear] = item;
    if (queue->front == -1) queue->front = queue->rear;
}

// Remove an item from the queue
struct Node* dequeue(struct Queue* queue) {
    if (is_empty(queue)) return NULL;
    struct Node* item = queue->array[queue->front];
    if (queue->front == queue->rear) queue->front = queue->rear = -1;
    else queue->front = (queue->front + 1) % queue->capacity;
    return item;
}

// BFS algorithm
void bfs(struct Node* start) {
    struct Queue* queue = create_queue(1000);
    start->visited = true;
    enqueue(queue, start);

    while (!is_empty(queue)) {
        struct Node* current = dequeue(queue);
        printf("%d ", current->value);

        for (int i = 0; i < current->num_neighbors; i++) {
            if (!current->neighbors[i]->visited) {
                current->neighbors[i]->visited = true;
                enqueue(queue, current->neighbors[i]);
            }
        }
    }

    free(queue->array);
    free(queue);
}

// DFS algorithm
void dfs(struct Node* node) {
    node->visited = true;
    printf("%d ", node->value);

    for (int i = 0; i < node->num_neighbors; i++) {
        if (!node->neighbors[i]->visited) {
            dfs(node->neighbors[i]);
        }
    }
}

int main() {
    // Create a simple neural network with one hidden layer
    int input_size = 3;
    int hidden_size = 4;
    int output_size = 2;

    struct Layer* hidden_layer = create_layer(hidden_size, input_size);
    struct Layer* output_layer = create_layer(output_size, hidden_size);

    // Example input
    double input[3] = {0.5, 0.3, 0.7};

    // Forward pass
    double* hidden_output = (double*)malloc(hidden_size * sizeof(double));
    layer_forward(hidden_layer, input, hidden_output);

    double* final_output = (double*)malloc(output_size * sizeof(double));
    layer_forward(output_layer, hidden_output, final_output);

    // Print neural network results
    printf("Neural Network:\n");
    printf("Input: [%.2f, %.2f, %.2f]\n", input[0], input[1], input[2]);
    printf("Output: [%.2f, %.2f]\n\n", final_output[0], final_output[1]);

    // Create a graph for search algorithms
    struct Node* node1 = create_node(1);
    struct Node* node2 = create_node(2);
    struct Node* node3 = create_node(3);
    struct Node* node4 = create_node(4);
    struct Node* node5 = create_node(5);

    add_edge(node1, node2);
    add_edge(node1, node3);
    add_edge(node2, node4);
    add_edge(node3, node4);
    add_edge(node3, node5);

    // Perform BFS
    printf("BFS traversal: ");
    bfs(node1);
    printf("\n");

    // Reset visited flags
    node1->visited = node2->visited = node3->visited = node4->visited = node5->visited = false;

    // Perform DFS
    printf("DFS traversal: ");
    dfs(node1);
    printf("\n");

    // Free allocated memory
    free_layer(hidden_layer);
    free_layer(output_layer);
    free(hidden_output);
    free(final_output);

    free(node1->neighbors);
    free(node2->neighbors);
    free(node3->neighbors);
    free(node4->neighbors);
    free(node5->neighbors);
    free(node1);
    free(node2);
    free(node3);
    free(node4);
    free(node5);

    return 0;
}
