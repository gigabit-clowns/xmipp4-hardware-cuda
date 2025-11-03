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

class cuda_default_buffer final
    : public cuda_buffer
{
public:
    cuda_default_buffer(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue, 
        cuda_memory_allocator &allocator
    );
    cuda_default_buffer(const cuda_default_buffer &other) = delete;
    cuda_default_buffer(cuda_default_buffer &&other) noexcept;
    ~cuda_default_buffer() override;

    cuda_default_buffer& operator=(const cuda_default_buffer &other) = delete;
    cuda_default_buffer& operator=(cuda_default_buffer &&other) noexcept;

    void reset() noexcept;
    void swap(cuda_default_buffer &other) noexcept;

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

}; 

} // namespace hardware
} // namespace xmipp4

