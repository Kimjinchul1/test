#include <stdio.h>   // 표준 입출력 함수를 사용하기 위한 헤더 파일
#include <stdlib.h>  // 동적 메모리 할당 함수를 사용하기 위한 헤더 파일
#include <math.h>    // 수학 함수를 사용하기 위한 헤더 파일 (현재 코드에서는 사용되지 않음)
#include <stdbool.h> // bool 자료형을 사용하기 위한 헤더 파일

// 뉴런 구조체 정의
// 뉴런은 인공 신경망의 기본 단위입니다. 각 뉴런은 여러 개의 입력을 받아 하나의 출력을 만듭니다.
struct Neuron {
    double* weights;  // 가중치 배열: 각 입력의 중요도를 나타내는 값들
    double bias;      // 편향: 뉴런의 활성화 임계값을 조절하는 값
    int num_inputs;   // 입력의 수: 이 뉴런이 받는 입력의 개수
};

// 뉴런 층 구조체 정의
// 층은 여러 개의 뉴런으로 구성됩니다. 인공 신경망은 보통 여러 개의 층으로 이루어집니다.
struct Layer {
    struct Neuron* neurons;  // 뉴런 배열: 이 층에 속한 모든 뉴런들
    int num_neurons;         // 뉴런의 수: 이 층에 있는 뉴런의 개수
};

// 그래프 노드 구조체 정의 (BFS와 DFS 알고리즘에 사용)
// 그래프는 여러 개의 노드와 그 노드들을 연결하는 간선으로 구성됩니다.
struct Node {
    int value;               // 노드의 값: 이 노드가 가지고 있는 데이터
    struct Node** neighbors; // 이웃 노드들의 배열: 이 노드와 직접 연결된 다른 노드들
    int num_neighbors;       // 이웃 노드의 수: 직접 연결된 노드의 개수
    bool visited;            // 방문 여부: 그래프 탐색 시 이 노드를 방문했는지 표시
};

// 새로운 뉴런을 생성하는 함수
// 동적 메모리 할당을 사용하여 뉴런을 생성하고 초기화합니다.
struct Neuron* create_neuron(int num_inputs) {
    // 뉴런 구조체를 위한 메모리 할당
    struct Neuron* neuron = (struct Neuron*)malloc(sizeof(struct Neuron));
    neuron->num_inputs = num_inputs;
    // 가중치 배열을 위한 메모리 할당
    neuron->weights = (double*)malloc(num_inputs * sizeof(double));
    // 편향을 -1에서 1 사이의 랜덤한 값으로 초기화
    neuron->bias = ((double)rand() / RAND_MAX) * 2 - 1;

    // 각 입력에 대한 가중치를 -1에서 1 사이의 랜덤한 값으로 초기화
    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    return neuron;
}

// 새로운 층을 생성하는 함수
// 지정된 수의 뉴런을 가진 새로운 층을 생성하고 초기화합니다.
struct Layer* create_layer(int num_neurons, int num_inputs) {
    // 층 구조체를 위한 메모리 할당
    struct Layer* layer = (struct Layer*)malloc(sizeof(struct Layer));
    layer->num_neurons = num_neurons;
    // 뉴런 배열을 위한 메모리 할당
    layer->neurons = (struct Neuron*)malloc(num_neurons * sizeof(struct Neuron));

    // 층의 각 뉴런을 생성하고 초기화
    for (int i = 0; i < num_neurons; i++) {
        layer->neurons[i] = *create_neuron(num_inputs);
    }

    return layer;
}

// 활성화 함수 (ReLU: Rectified Linear Unit)
// ReLU는 입력이 0보다 크면 그대로 출력하고, 0 이하면 0을 출력합니다.
// 이 함수는 뉴런의 출력을 비선형으로 만들어 신경망이 복잡한 패턴을 학습할 수 있게 합니다.
double relu(double x) {
    return x > 0 ? x : 0;
}

// 단일 뉴런의 순전파 (forward pass) 함수
// 입력값들과 가중치를 곱하고 편향을 더한 후 활성화 함수를 적용합니다.
double neuron_forward(struct Neuron* neuron, double* inputs) {
    double sum = neuron->bias;
    for (int i = 0; i < neuron->num_inputs; i++) {
        sum += neuron->weights[i] * inputs[i];
    }
    return relu(sum);
}

// 층의 순전파 함수
// 층에 있는 모든 뉴런에 대해 순전파를 수행합니다.
void layer_forward(struct Layer* layer, double* inputs, double* outputs) {
    for (int i = 0; i < layer->num_neurons; i++) {
        outputs[i] = neuron_forward(&layer->neurons[i], inputs);
    }
}

// 뉴런에 할당된 메모리를 해제하는 함수
// 동적으로 할당된 메모리를 해제하여 메모리 누수를 방지합니다.
void free_neuron(struct Neuron* neuron) {
    free(neuron->weights);
    free(neuron);
}

// 층에 할당된 메모리를 해제하는 함수
// 층에 속한 모든 뉴런의 메모리를 해제하고, 층 자체의 메모리도 해제합니다.
void free_layer(struct Layer* layer) {
    for (int i = 0; i < layer->num_neurons; i++) {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
    free(layer);
}

// 그래프 노드를 생성하는 함수
// 새로운 노드를 생성하고 초기화합니다.
struct Node* create_node(int value) {
    struct Node* node = (struct Node*)malloc(sizeof(struct Node));
    node->value = value;
    node->neighbors = NULL;
    node->num_neighbors = 0;
    node->visited = false;
    return node;
}

// 그래프에 엣지(간선)를 추가하는 함수
// 두 노드를 서로 연결합니다. 무방향 그래프를 가정하므로 양방향으로 연결됩니다.
void add_edge(struct Node* node1, struct Node* node2) {
    // 각 노드의 이웃 수를 증가시킵니다.
    node1->num_neighbors++;
    node2->num_neighbors++;
    // 각 노드의 이웃 배열을 확장합니다.
    node1->neighbors = (struct Node**)realloc(node1->neighbors, node1->num_neighbors * sizeof(struct Node*));
    node2->neighbors = (struct Node**)realloc(node2->neighbors, node2->num_neighbors * sizeof(struct Node*));
    // 서로를 이웃으로 추가합니다.
    node1->neighbors[node1->num_neighbors - 1] = node2;
    node2->neighbors[node2->num_neighbors - 1] = node1;
}

// BFS(너비 우선 탐색)에 사용할 큐 구조체
// 큐는 선입선출(FIFO) 방식으로 동작하는 자료구조입니다.
struct Queue {
    struct Node** array;  // 노드 포인터의 배열
    int front;            // 큐의 앞부분 (데이터를 꺼내는 위치)
    int rear;             // 큐의 뒷부분 (데이터를 넣는 위치)
    int capacity;         // 큐의 최대 용량
};

// 큐를 생성하는 함수
struct Queue* create_queue(int capacity) {
    struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
    queue->capacity = capacity;
    queue->front = queue->rear = -1;
    queue->array = (struct Node**)malloc(queue->capacity * sizeof(struct Node*));
    return queue;
}

// 큐가 가득 찼는지 확인하는 함수
int is_full(struct Queue* queue) {
    return ((queue->rear + 1) % queue->capacity == queue->front);
}

// 큐가 비어있는지 확인하는 함수
int is_empty(struct Queue* queue) {
    return (queue->front == -1);
}

// 큐에 항목을 추가하는 함수 (enqueue)
void enqueue(struct Queue* queue, struct Node* item) {
    if (is_full(queue)) return;
    queue->rear = (queue->rear + 1) % queue->capacity;
    queue->array[queue->rear] = item;
    if (queue->front == -1) queue->front = queue->rear;
}

// 큐에서 항목을 제거하고 반환하는 함수 (dequeue)
struct Node* dequeue(struct Queue* queue) {
    if (is_empty(queue)) return NULL;
    struct Node* item = queue->array[queue->front];
    if (queue->front == queue->rear) queue->front = queue->rear = -1;
    else queue->front = (queue->front + 1) % queue->capacity;
    return item;
}

// BFS(너비 우선 탐색) 알고리즘
// 시작 노드부터 시작하여 가까운 노드부터 탐색하는 알고리즘입니다.
void bfs(struct Node* start) {
    struct Queue* queue = create_queue(1000);
    start->visited = true;
    enqueue(queue, start);

    while (!is_empty(queue)) {
        struct Node* current = dequeue(queue);
        printf("%d ", current->value);  // 현재 노드의 값을 출력

        // 현재 노드의 모든 이웃을 검사
        for (int i = 0; i < current->num_neighbors; i++) {
            if (!current->neighbors[i]->visited) {
                current->neighbors[i]->visited = true;
                enqueue(queue, current->neighbors[i]);
            }
        }
    }

    // 사용이 끝난 큐의 메모리를 해제
    free(queue->array);
    free(queue);
}

// DFS(깊이 우선 탐색) 알고리즘
// 시작 노드에서 가능한 깊이 들어가면서 탐색하는 알고리즘입니다.
void dfs(struct Node* node) {
    node->visited = true;
    printf("%d ", node->value);  // 현재 노드의 값을 출력

    // 현재 노드의 모든 이웃에 대해 재귀적으로 DFS 수행
    for (int i = 0; i < node->num_neighbors; i++) {
        if (!node->neighbors[i]->visited) {
            dfs(node->neighbors[i]);
        }
    }
}

int main() {
    // 하나의 은닉층을 가진 간단한 신경망 생성
    int input_size = 3;   // 입력층의 뉴런 수
    int hidden_size = 4;  // 은닉층의 뉴런 수
    int output_size = 2;  // 출력층의 뉴런 수

    // 은닉층과 출력층 생성
    struct Layer* hidden_layer = create_layer(hidden_size, input_size);
    struct Layer* output_layer = create_layer(output_size, hidden_size);

    // 예제 입력 데이터
    double input[3] = {0.5, 0.3, 0.7};

    // 신경망의 순전파 수행
    // 1. 입력층에서 은닉층으로
    double* hidden_output = (double*)malloc(hidden_size * sizeof(double));
    layer_forward(hidden_layer, input, hidden_output);

    // 2. 은닉층에서 출력층으로
    double* final_output = (double*)malloc(output_size * sizeof(double));
    layer_forward(output_layer, hidden_output, final_output);

    // 신경망의 결과 출력
    printf("신경망:\n");
    printf("입력: [%.2f, %.2f, %.2f]\n", input[0], input[1], input[2]);
    printf("출력: [%.2f, %.2f]\n\n", final_output[0], final_output[1]);

    // 그래프 생성 (BFS와 DFS
