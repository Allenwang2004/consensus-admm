#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>

struct SystemResources {
    size_t total_memory_bytes;
    size_t available_memory_bytes;
    size_t free_memory_bytes;
    size_t inactive_memory_bytes;
    size_t cpu_cores;
    
    SystemResources() {
        cpu_cores = static_cast<size_t>(std::thread::hardware_concurrency());
        
        // 获取总内存
        int mib[2];
        int64_t physical_memory;
        size_t length;
        mib[0] = CTL_HW;
        mib[1] = HW_MEMSIZE;
        length = sizeof(int64_t);
        sysctl(mib, 2, &physical_memory, &length, NULL, 0);
        total_memory_bytes = physical_memory;
        
        // 获取实时内存使用情况
        update_memory_info();
    }
    
    void update_memory_info() {
        vm_size_t page_size;
        vm_statistics64_data_t vm_stat;
        mach_msg_type_number_t host_size = sizeof(vm_statistics64_data_t) / sizeof(natural_t);
        
        host_page_size(mach_host_self(), &page_size);
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size);
        
        // 计算各种内存状态（以字节为单位）
        free_memory_bytes = vm_stat.free_count * page_size;
        inactive_memory_bytes = vm_stat.inactive_count * page_size;
        
        // 可用内存 = 空闲内存 + 非活跃内存（可以被回收）
        available_memory_bytes = free_memory_bytes + inactive_memory_bytes;
        
        // 为安全起见，预留一些内存给系统
        size_t safety_buffer = total_memory_bytes * 0.1; // 预留10%
        if (available_memory_bytes > safety_buffer) {
            available_memory_bytes -= safety_buffer;
        } else {
            available_memory_bytes = available_memory_bytes * 0.5; // 如果可用内存很少，只用一半
        }
    }
    
    void print_info() const {
        std::cout << "=== System Resources ===" << std::endl;
        std::cout << "CPU cores: " << cpu_cores << std::endl;
        std::cout << "Total RAM: " << total_memory_bytes / (1024.0 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "Free RAM: " << free_memory_bytes / (1024.0 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "Inactive RAM: " << inactive_memory_bytes / (1024.0 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "Available RAM (estimated): " << available_memory_bytes / (1024.0 * 1024 * 1024) << " GB" << std::endl;
        
        // 计算内存使用率
        double memory_usage = (total_memory_bytes - free_memory_bytes - inactive_memory_bytes) / (double)total_memory_bytes * 100;
        std::cout << "Memory usage: " << memory_usage << "%" << std::endl;
    }
    
    // 获取更详细的内存信息
    void print_detailed_memory_info() const {
        vm_size_t page_size;
        vm_statistics64_data_t vm_stat;
        mach_msg_type_number_t host_size = sizeof(vm_statistics64_data_t) / sizeof(natural_t);
        
        host_page_size(mach_host_self(), &page_size);
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size);
        
        std::cout << "\n=== Detailed Memory Statistics ===" << std::endl;
        std::cout << "Page size: " << page_size << " bytes" << std::endl;
        std::cout << "Free pages: " << vm_stat.free_count << " (" << (vm_stat.free_count * page_size) / (1024.0 * 1024) << " MB)" << std::endl;
        std::cout << "Active pages: " << vm_stat.active_count << " (" << (vm_stat.active_count * page_size) / (1024.0 * 1024) << " MB)" << std::endl;
        std::cout << "Inactive pages: " << vm_stat.inactive_count << " (" << (vm_stat.inactive_count * page_size) / (1024.0 * 1024) << " MB)" << std::endl;
        std::cout << "Wired pages: " << vm_stat.wire_count << " (" << (vm_stat.wire_count * page_size) / (1024.0 * 1024) << " MB)" << std::endl;
        std::cout << "Compressed pages: " << vm_stat.compressor_page_count << " (" << (vm_stat.compressor_page_count * page_size) / (1024.0 * 1024) << " MB)" << std::endl;
    }
};

struct TaskConfiguration {
    size_t num_threads;
    size_t chunk_size;
    size_t memory_per_thread;
    size_t total_batches; // 新增：如果需要分批处理
    
    void print_config() const {
        std::cout << "=== Task Configuration ===" << std::endl;
        std::cout << "Threads: " << num_threads << std::endl;
        std::cout << "Chunk size per thread: " << chunk_size << std::endl;
        std::cout << "Memory per thread: " << memory_per_thread / (1024.0 * 1024) << " MB" << std::endl;
        if (total_batches > 1) {
            std::cout << "Total batches needed: " << total_batches << std::endl;
        }
        std::cout << "Total memory usage: " << (num_threads * memory_per_thread) / (1024.0 * 1024) << " MB" << std::endl;
    }
};

class ResourceManager {
private:
    SystemResources resources;
    
public:
    // 刷新系统资源信息
    void refresh_resources() {
        resources.update_memory_info();
    }
    
    TaskConfiguration optimize_task_allocation(size_t total_data_size, size_t bytes_per_element) {
        // 获取最新的内存信息
        resources.update_memory_info();
        
        TaskConfiguration config;
        config.total_batches = 1;
        
        // 计算总数据的内存需求
        size_t total_memory_needed = total_data_size * bytes_per_element;
        
        std::cout << "\n=== Memory Analysis ===" << std::endl;
        std::cout << "Total data size: " << total_data_size << " elements" << std::endl;
        std::cout << "Memory needed: " << total_memory_needed / (1024.0 * 1024) << " MB" << std::endl;
        std::cout << "Available memory: " << resources.available_memory_bytes / (1024.0 * 1024) << " MB" << std::endl;
        
        if (total_memory_needed <= resources.available_memory_bytes) {
            // 内存充足，可以全部加载
            config.num_threads = std::min(resources.cpu_cores, total_data_size);
            config.chunk_size = std::max(static_cast<size_t>(1), total_data_size / config.num_threads);
            config.memory_per_thread = (total_memory_needed + config.num_threads - 1) / config.num_threads;
            
            std::cout << "Strategy: Load all data into memory" << std::endl;
        } else {
            // 内存不足，需要分批或优化处理
            size_t max_elements_in_memory = resources.available_memory_bytes / bytes_per_element;
            
            if (max_elements_in_memory < resources.cpu_cores) {
                // 内存严重不足
                config.num_threads = 1;
                config.chunk_size = max_elements_in_memory / 2; // 保守一点
                config.memory_per_thread = config.chunk_size * bytes_per_element;
                config.total_batches = (total_data_size + config.chunk_size - 1) / config.chunk_size;
                
                std::cout << "Strategy: Sequential processing with small chunks" << std::endl;
            } else {
                // 可以并行，但需要分批
                config.num_threads = resources.cpu_cores;
                config.chunk_size = max_elements_in_memory / config.num_threads;
                config.memory_per_thread = config.chunk_size * bytes_per_element;
                
                size_t elements_per_batch = config.num_threads * config.chunk_size;
                config.total_batches = (total_data_size + elements_per_batch - 1) / elements_per_batch;
                
                std::cout << "Strategy: Parallel processing with batching" << std::endl;
            }
        }
        
        return config;
    }
    
    TaskConfiguration adaptive_allocation(size_t total_data_size, 
                                        size_t bytes_per_element,
                                        double cpu_intensity_factor = 1.0,
                                        double memory_safety_factor = 0.8) {
        resources.update_memory_info();
        
        TaskConfiguration config;
        config.total_batches = 1;
        
        size_t safe_memory = static_cast<size_t>(resources.available_memory_bytes * memory_safety_factor);
        size_t total_memory_needed = total_data_size * bytes_per_element;
        
        // 根据计算密集度调整线程数
        size_t optimal_threads = static_cast<size_t>(std::ceil(resources.cpu_cores * cpu_intensity_factor));
        optimal_threads = std::min(optimal_threads, total_data_size);
        
        if (total_memory_needed <= safe_memory) {
            // 内存充足的情况
            config.num_threads = optimal_threads;
            config.chunk_size = (total_data_size + config.num_threads - 1) / config.num_threads;
            config.memory_per_thread = (total_memory_needed + config.num_threads - 1) / config.num_threads;
        } else {
            // 内存受限的情况
            size_t max_elements_per_batch = safe_memory / bytes_per_element;
            
            if (max_elements_per_batch >= total_data_size) {
                config.num_threads = optimal_threads;
                config.chunk_size = (total_data_size + config.num_threads - 1) / config.num_threads;
                config.memory_per_thread = config.chunk_size * bytes_per_element;
            } else {
                // 需要分批处理
                config.num_threads = std::min(optimal_threads, max_elements_per_batch);
                config.chunk_size = max_elements_per_batch / config.num_threads;
                config.memory_per_thread = config.chunk_size * bytes_per_element;
                
                size_t elements_per_batch = config.num_threads * config.chunk_size;
                config.total_batches = (total_data_size + elements_per_batch - 1) / elements_per_batch;
            }
        }
        
        return config;
    }
    
    void print_system_info() {
        resources.print_info();
    }
    
    void print_detailed_info() {
        resources.print_detailed_memory_info();
    }
};

// 示例使用
int main() {
    ResourceManager manager;
    manager.print_system_info();
    manager.print_detailed_info();
    
    // 示例：处理1000万个双精度浮点数
    size_t total_elements = 10000000;
    size_t bytes_per_double = sizeof(double);
    
    std::cout << "\n=== Basic Optimization ===" << std::endl;
    TaskConfiguration config1 = manager.optimize_task_allocation(total_elements, bytes_per_double);
    config1.print_config();
    
    std::cout << "\n=== Adaptive Optimization (CPU intensive) ===" << std::endl;
    TaskConfiguration config2 = manager.adaptive_allocation(total_elements, bytes_per_double, 1.5, 0.7);
    config2.print_config();
    
    std::cout << "\n=== Adaptive Optimization (Memory intensive) ===" << std::endl;
    TaskConfiguration config3 = manager.adaptive_allocation(total_elements, bytes_per_double, 0.8, 0.9);
    config3.print_config();
    
    return 0;
}
#endif