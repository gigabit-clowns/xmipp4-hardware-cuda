// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_allocator.hpp"

#include "cuda_buffer_implementation.hpp"

namespace xmipp4
{
namespace hardware
{

XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_MEMORY_REQUEST_ROUND_STEP = 512;
XMIPP4_CONST_CONSTEXPR
std::size_t XMIPP4_CUDA_MEMORY_ALLOCATE_ROUND_STEP = 2<<20; // 2MB

cuda_memory_allocator::cuda_memory_allocator(cuda_memory_resource &resource)
    : m_resource(resource)
    , m_cache(
        XMIPP4_CUDA_MEMORY_REQUEST_ROUND_STEP, 
        XMIPP4_CUDA_MEMORY_ALLOCATE_ROUND_STEP
    ) 
{
}

cuda_memory_resource& cuda_memory_allocator::get_memory_resource() const noexcept
{
    return m_resource;
}

std::shared_ptr<buffer> cuda_memory_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    device_queue *queue
)
{
    cuda_device_queue *cuda_queue = nullptr;
    if (queue)
    {
        cuda_queue = &dynamic_cast<cuda_device_queue&>(*queue);
    }

    return allocate(size, alignment, cuda_queue);
}

std::shared_ptr<cuda_buffer> cuda_memory_allocator::allocate(
    std::size_t size, 
    std::size_t alignment, 
    cuda_device_queue *queue
)
{
    return std::make_shared<cuda_buffer_implementation>(
        size, 
        alignment, 
        queue, 
        *this
    );
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
