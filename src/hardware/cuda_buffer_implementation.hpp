// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/cuda/hardware/cuda_buffer.hpp>

#include "cuda_memory_allocator.hpp"

#include <utility>

namespace xmipp4 
{
namespace hardware
{

class cuda_device_queue;
class cuda_memory_allocator;
class cuda_memory_block;
class cuda_memory_block_usage_tracker;

class cuda_buffer_implementation final
    : public cuda_buffer
{
public:
    cuda_buffer_implementation(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue, 
        cuda_memory_allocator &allocator
    );
    cuda_buffer_implementation(const cuda_buffer_implementation &other) = delete;
    cuda_buffer_implementation(cuda_buffer_implementation &&other) noexcept;
    ~cuda_buffer_implementation() override;

    cuda_buffer_implementation& 
    operator=(const cuda_buffer_implementation &other) = delete;
    cuda_buffer_implementation& 
    operator=(cuda_buffer_implementation &&other) noexcept;

    void reset() noexcept;
    void swap(cuda_buffer_implementation &other) noexcept;

    std::size_t get_size() const noexcept override;

    memory_resource& get_memory_resource() const noexcept override;

    void record_queue(cuda_device_queue &queue, bool exclusive=false) override;

    void* get_host_ptr() noexcept override;

    const void* get_host_ptr() const noexcept override;

    void* get_device_ptr() noexcept;

    const void* get_device_ptr() const noexcept;

private:
    std::reference_wrapper<cuda_memory_allocator> m_allocator;
    const cuda_memory_block *m_block;
    cuda_memory_block_usage_tracker *m_tracker;
    void *m_host_ptr;
    void *m_device_ptr;

}; 

} // namespace hardware
} // namespace xmipp4
