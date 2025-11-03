// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_default_buffer.hpp"

#include "cuda_memory_allocator.hpp"

namespace xmipp4
{
namespace hardware
{

cuda_default_buffer::cuda_default_buffer(
    std::size_t size, 
    std::size_t alignment, 
    cuda_device_queue *queue, 
    cuda_memory_allocator &allocator
)
    : m_allocator(allocator)
{
    auto &block = allocator.allocate_block(
        size,
        alignment,
        queue,
        &m_tracker
    );
    m_block = &block;
}

cuda_default_buffer::cuda_default_buffer(cuda_default_buffer &&other) noexcept
    : m_allocator(other.m_allocator)
    , m_block(nullptr)
    , m_tracker(nullptr)
{
    swap(other);
}

cuda_default_buffer::~cuda_default_buffer()
{
    reset();
}

cuda_default_buffer& cuda_default_buffer::operator=(cuda_default_buffer &&other) noexcept
{
    swap(other);
    other.reset();
    return *this;
}

void cuda_default_buffer::reset() noexcept
{
    if (m_block)
    {
        m_allocator.get().deallocate_block(*m_block);
        m_block = nullptr;
        m_tracker = nullptr;
    }
}

void cuda_default_buffer::swap(cuda_default_buffer &other) noexcept
{
    std::swap(m_allocator, other.m_allocator);
    std::swap(m_block, other.m_block);
    std::swap(m_tracker, other.m_tracker);
}

void* cuda_default_buffer::get_host_ptr() noexcept
{
    // TODO
}

const void* cuda_default_buffer::get_host_ptr() const noexcept
{
    // TODO
}

std::size_t cuda_default_buffer::get_size() const noexcept
{
    if (m_block)
    {
        return m_block->get_size();
    }
    else
    {
        return 0UL;
    }
}

memory_resource& cuda_default_buffer::get_memory_resource() const noexcept
{
    return m_allocator.get().get_memory_resource();
}

void cuda_default_buffer::record_queue(cuda_device_queue &queue, bool)
{
    if (m_block)
    {
        XMIPP4_ASSERT(m_tracker);
        m_tracker->add_queue(*m_block, queue);
    }
}

void* cuda_default_buffer::get_device_ptr() noexcept
{
    // TODO
}

const void* cuda_default_buffer::get_device_ptr() const noexcept
{
    // TODO
}

} // namespace hardware
} // namespace xmipp4
