// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "cuda_memory_block_allocator.hpp"

#include <utility>

namespace xmipp4 
{
namespace hardware
{

class cuda_device_queue;
class cuda_memory_allocator;
class cuda_memory_block;
class cuda_memory_block_usage_tracker;

class cuda_memory_block_allocation
{
public:
    cuda_memory_block_allocation(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue, 
        cuda_memory_block_allocator &allocator
    );
    cuda_memory_block_allocation(
        const cuda_memory_block_allocation &other
    ) = delete;
    cuda_memory_block_allocation(
        cuda_memory_block_allocation &&other
    ) noexcept;
    ~cuda_memory_block_allocation();

    cuda_memory_block_allocation& 
    operator=(const cuda_memory_block_allocation &other) = delete;
    cuda_memory_block_allocation& 
    operator=(cuda_memory_block_allocation &&other) noexcept;

    void reset() noexcept;
    void swap(cuda_memory_block_allocation &other) noexcept;

    std::size_t get_size() const noexcept;

    cuda_memory_block_allocator& get_allocator() const noexcept;

    void record_queue(cuda_device_queue &queue);

    const cuda_memory_block* get_memory_block() const noexcept;

private:
    std::reference_wrapper<cuda_memory_block_allocator> m_allocator;
    const cuda_memory_block *m_block;
    cuda_memory_block_usage_tracker *m_tracker;

}; 

} // namespace hardware
} // namespace xmipp4
