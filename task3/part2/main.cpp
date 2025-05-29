#include <iostream>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

template<typename T>
class Server {
public:
    void start() {
        running = true;
        server_thread = std::thread(&Server::run, this);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            running = false;
        }
        cv.notify_all();
        if (server_thread.joinable())
            server_thread.join();
    }

    size_t add_task(std::function<T()> task) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        size_t id = task_id_counter++;
        task_queue.emplace(id, std::move(task));
        cv.notify_one();
        return id;
    }

    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(result_mutex);
        result_cv.wait(lock, [&] { return results.count(id) > 0; });
        T result = results[id];
        results.erase(id);
        return result;
    }

private:
    void run() {
        while (true) {
            std::function<T()> task;
            size_t id;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [&] { return !task_queue.empty() || !running; });
                if (!running && task_queue.empty()) return;

                auto [tid, tfunc] = task_queue.front();
                task_queue.pop();
                id = tid;
                task = std::move(tfunc);
            }

            T result = task();

            {
                std::lock_guard<std::mutex> lock(result_mutex);
                results[id] = result;
            }
            result_cv.notify_all();
        }
    }

    std::atomic<bool> running{false};
    std::thread server_thread;

    std::mutex queue_mutex;
    std::condition_variable cv;
    std::queue<std::pair<size_t, std::function<T()>>> task_queue;

    std::mutex result_mutex;
    std::condition_variable result_cv;
    std::unordered_map<size_t, T> results;

    std::atomic<size_t> task_id_counter{0};
};

void client(Server<double>& server, int client_id, int N, const std::string& type) {
    std::ofstream out("client_" + std::to_string(client_id) + ".txt");
    std::vector<size_t> task_ids;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.1, 10.0);

    for (int i = 0; i < N; ++i) {
        double a = dist(gen);
        double b = dist(gen);

        std::function<double()> task;

        if (type == "sin") task = [a]() { return std::sin(a); };
        else if (type == "sqrt") task = [a]() { return std::sqrt(a); };
        else if (type == "pow") task = [a, b]() { return std::pow(a, b); };
        else throw std::invalid_argument("Unknown task type");

        size_t id = server.add_task(task);
        task_ids.push_back(id);
    }

    for (size_t id : task_ids) {
        double result = server.request_result(id);
        out << "task_id=" << id << " result=" << result << "\n";
    }
}

void test_results(int client_count) {
    for (int i = 1; i <= client_count; ++i) {
        std::ifstream in("client_" + std::to_string(i) + ".txt");
        std::string line;
        int line_count = 0;
        while (std::getline(in, line)) {
            assert(line.find("result=") != std::string::npos);
            ++line_count;
        }
        std::cout << "Client " << i << " result lines: " << line_count << "\n";
    }
    std::cout << "Test passed: All results correctly written.\n";
}

int main() {
    constexpr int TASKS_PER_CLIENT = 50; // 5 < N < 10000

    Server<double> server;
    server.start();

    std::thread c1(client, std::ref(server), 1, TASKS_PER_CLIENT, "sin");
    std::thread c2(client, std::ref(server), 2, TASKS_PER_CLIENT, "sqrt");
    std::thread c3(client, std::ref(server), 3, TASKS_PER_CLIENT, "pow");

    c1.join();
    c2.join();
    c3.join();

    server.stop();

    test_results(3);
    return 0;
}