// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_buffer_implementation.hpp"

#include "cuda_memory_allocator.hpp"

namespace xmipp4
{
namespace hardware
{

cuda_buffer_implementation::cuda_buffer_implementation(
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

cuda_buffer_implementation::cuda_buffer_implementation(cuda_buffer_implementation &&other) noexcept
    : m_allocator(other.m_allocator)
    , m_block(nullptr)
    , m_tracker(nullptr)
    , m_host_ptr(nullptr)
    , m_device_ptr(nullptr)
{
    swap(other);
}

cuda_buffer_implementation::~cuda_buffer_implementation()
{
    reset();
}

cuda_buffer_implementation& cuda_buffer_implementation::operator=(cuda_buffer_implementation &&other) noexcept
{
    swap(other);
    other.reset();
    return *this;
}

void cuda_buffer_implementation::reset() noexcept
{
    if (m_block)
    {
        m_allocator.get().deallocate_block(*m_block);
        m_block = nullptr;
        m_tracker = nullptr;
        m_host_ptr = nullptr;
        m_device_ptr = nullptr;
    }
}

void cuda_buffer_implementation::swap(cuda_buffer_implementation &other) noexcept
{
    std::swap(m_allocator, other.m_allocator);
    std::swap(m_block, other.m_block);
    std::swap(m_tracker, other.m_tracker);
    std::swap(m_host_ptr, other.m_host_ptr);
    std::swap(m_device_ptr, other.m_device_ptr);
}

void* cuda_buffer_implementation::get_host_ptr() noexcept
{
    return m_host_ptr;
}

const void* cuda_buffer_implementation::get_host_ptr() const noexcept
{
    return m_host_ptr;
}

std::size_t cuda_buffer_implementation::get_size() const noexcept
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

memory_resource& cuda_buffer_implementation::get_memory_resource() const noexcept
{
    return m_allocator.get().get_memory_resource();
}

void cuda_buffer_implementation::record_queue(cuda_device_queue &queue, bool)
{
    if (m_block)
    {
        XMIPP4_ASSERT(m_tracker);
        m_tracker->add_queue(*m_block, queue);
    }
}

void* cuda_buffer_implementation::get_device_ptr() noexcept
{
    return m_device_ptr;
}

const void* cuda_buffer_implementation::get_device_ptr() const noexcept
{
    return m_device_ptr;
}

} // namespace hardware
} // namespace xmipp4
