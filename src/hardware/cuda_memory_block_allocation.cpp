// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_block_allocation.hpp"

namespace xmipp4
{
namespace hardware
{

cuda_memory_block_allocation::cuda_memory_block_allocation(
    std::size_t size, 
    std::size_t alignment, 
    cuda_device_queue *queue, 
    cuda_memory_block_allocator &allocator
)
    : m_allocator(allocator)
{
    auto &block = allocator.allocate(
        size,
        alignment,
        queue,
        &m_tracker
    );
    m_block = &block;
}

cuda_memory_block_allocation::cuda_memory_block_allocation(
    cuda_memory_block_allocation &&other
) noexcept
    : m_allocator(other.m_allocator)
    , m_block(nullptr)
    , m_tracker(nullptr)
{
    swap(other);
}

cuda_memory_block_allocation::~cuda_memory_block_allocation()
{
    reset();
}

cuda_memory_block_allocation& cuda_memory_block_allocation::operator=(
    cuda_memory_block_allocation &&other
) noexcept
{
    swap(other);
    other.reset();
    return *this;
}

void cuda_memory_block_allocation::reset() noexcept
{
    if (m_block)
    {
        get_allocator().deallocate(*m_block);
        m_block = nullptr;
        m_tracker = nullptr;
    }
}

void cuda_memory_block_allocation::swap(cuda_memory_block_allocation &other) noexcept
{
    std::swap(m_allocator, other.m_allocator);
    std::swap(m_block, other.m_block);
    std::swap(m_tracker, other.m_tracker);
}

cuda_memory_block_allocator& 
cuda_memory_block_allocation::get_allocator() const noexcept
{
    return m_allocator;
}

void cuda_memory_block_allocation::record_queue(cuda_device_queue &queue)
{
    if (m_block)
    {
        XMIPP4_ASSERT(m_tracker);
        m_tracker->add_queue(*m_block, queue);
    }
}

const cuda_memory_block* 
cuda_memory_block_allocation::get_memory_block() const noexcept
{
    return m_block;
}

} // namespace hardware
} // namespace xmipp4
