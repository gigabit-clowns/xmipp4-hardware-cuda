// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_allocator.hpp"

#include <xmipp4/cuda/hardware/cuda_buffer.hpp>

namespace xmipp4
{
namespace hardware
{

cuda_memory_allocator::cuda_memory_allocator(cuda_memory_resource &resource)
    : m_resource(resource)
    , m_cache()
{
}

memory_resource& cuda_memory_allocator::get_memory_resource() const noexcept
{
    return m_resource;
}

std::shared_ptr<buffer> cuda_memory_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    device_queue *queue
)
{
    return std::make_shared<cuda_buffer>(size, alignment, queue, *this);
}

void cuda_memory_allocator::release_free_blocks()
{
    m_cache.release(get_memory_resource());
}

const cuda_memory_block& cuda_memory_allocator::allocate_block(
    std::size_t size, 
    std::size_t alignment,
    const cuda_device_queue *queue,
    cuda_memory_block_usage_tracker **usage_tracker
)
{
    auto &resource = get_memory_resource();
    auto *result = m_cache.allocate(
        resource, 
        size, 
        alignment, 
        queue, 
        usage_tracker
    );

    if (!result)
    {
        // Retry after releasing blocks
        m_cache.release(resource);
        result = m_cache.allocate(
            resource, 
            size, 
            alignment, 
            queue, 
            usage_tracker
        );
    }

    if (!result)
    {
        throw std::bad_alloc();
    }

    return *result;
}

void cuda_memory_allocator::deallocate_block(const cuda_memory_block &block)
{
    m_cache.deallocate(block);
}

} // namespace hardware
} // namespace xmipp4
