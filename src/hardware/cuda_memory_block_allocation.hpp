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

/**
 * @brief RAII style class to represent an allocated cuda_memory_block.
 * 
 */
class cuda_memory_block_allocation
{
public:
    /**
     * @brief Construct a new cuda_memory_block_allocation.
     * 
     * @param size Minimum size requirement for the memory block.
     * @param alignment Minimum alignment requirement for the memory block.
     * @param queue Queue where the allocation must be available.
     * @param allocator Block allocator from which the memory is obtained.
     */
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

    /**
     * @brief Release the allocated memory block.
     * 
     */
    void reset() noexcept;

    /**
     * @brief Swap responsibilities with another memory block allocation.
     * 
     * @param other 
     */
    void swap(cuda_memory_block_allocation &other) noexcept;

    /**
     * @brief Get the allocator used for allocating this block.
     * 
     * @return cuda_memory_block_allocator& The allocator used
     * for allocating this block.
     */
    cuda_memory_block_allocator& get_allocator() const noexcept;

    /**
     * @brief Acknowledge that the block is being used with an additional queue.
     * 
     * @param queue The queue where the block is being used.
     */
    void record_queue(cuda_device_queue &queue);

    /**
     * @brief Get the memory block held by this allocation.
     * 
     * @return const cuda_memory_block* The memory block held in this 
     * allocation. nullptr if no value is held.
     */
    const cuda_memory_block* get_memory_block() const noexcept;

private:
    std::reference_wrapper<cuda_memory_block_allocator> m_allocator;
    const cuda_memory_block *m_block;
    cuda_memory_block_usage_tracker *m_tracker;

}; 

} // namespace hardware
} // namespace xmipp4
