// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_allocator.hpp>

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

#include "cuda_memory_block_cache.hpp"

namespace xmipp4 
{
namespace hardware
{

class cuda_device_queue;
class cuda_buffer;

class cuda_memory_allocator final
    : public memory_allocator
{
public:
    explicit cuda_memory_allocator(cuda_memory_resource &resource);
    cuda_memory_allocator(const cuda_memory_allocator &other) = delete;
    cuda_memory_allocator(cuda_memory_allocator &&other) = default;
    ~cuda_memory_allocator() override = default;

    cuda_memory_allocator& 
    operator=(const cuda_memory_allocator &other) = delete;
    cuda_memory_allocator& 
    operator=(cuda_memory_allocator &&other) = default;

    cuda_memory_resource& get_memory_resource() const noexcept override;

    std::shared_ptr<buffer> allocate(
        std::size_t size, 
        std::size_t alignment, 
        device_queue *queue
    ) override;

    std::shared_ptr<cuda_buffer> allocate(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue
    );

    /**
     * @brief Return free blocks to the memory resource when possible.
     * 
     */
    void release_free_blocks();

    /**
     * @brief Allocate a new block.
     * 
     * @param size Size of the requested block.
     * @param alignment Alignment requirement for the requested blocks.
     * @param queue Queue of the requested block.
     * @param usage_tracker Output parameter to register alien queues. May be 
     * nullptr. Ownership is managed by the allocator and the caller shall not
     * call any delete/free on it.
     * @return const cuda_memory_block&
     * 
     */
    const cuda_memory_block& allocate_block(
        std::size_t size, 
        std::size_t alignment,
        const cuda_device_queue *queue,
        cuda_memory_block_usage_tracker **usage_tracker = nullptr 
    );

    /**
     * @brief Deallocate a block.
     * 
     * @param block Block to be deallocated.
     * 
     * @note This operation does not return the block to the allocator.
     * Instead, it caches it for potential re-use.
     * 
     */
    void deallocate_block(const cuda_memory_block &block);

private:    
    std::reference_wrapper<cuda_memory_resource> m_resource;
    cuda_memory_block_cache m_cache;

}; 

} // namespace hardware
} // namespace xmipp4

