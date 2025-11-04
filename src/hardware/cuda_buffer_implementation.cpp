// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_buffer_implementation.hpp"

namespace xmipp4
{
namespace hardware
{

cuda_buffer_implementation::cuda_buffer_implementation(
    std::size_t size, 
    std::size_t alignment, 
    cuda_device_queue *queue, 
    cuda_memory_block_allocator &allocator
)
    : m_allocation(size, alignment, queue, allocator)
    , m_kind(m_allocation.get_allocator().get_memory_resource().get_kind())
{
}

void* cuda_buffer_implementation::get_device_ptr() noexcept
{
    return get_device_ptr_impl();
}

const void* cuda_buffer_implementation::get_device_ptr() const noexcept
{
    return get_device_ptr_impl();
}

void* cuda_buffer_implementation::get_host_ptr() noexcept
{
    return get_host_ptr_impl();
}

const void* cuda_buffer_implementation::get_host_ptr() const noexcept
{
    return get_host_ptr_impl();
}

std::size_t cuda_buffer_implementation::get_size() const noexcept
{
    auto *block = m_allocation.get_memory_block();
    if (block)
    {
        return block->get_size();
    }
    else
    {
        return 0UL;
    }
}

cuda_memory_resource& 
cuda_buffer_implementation::get_memory_resource() const noexcept
{
    return m_allocation.get_allocator().get_memory_resource();
}

void cuda_buffer_implementation::record_queue(device_queue &queue, bool)
{
    m_allocation.record_queue(dynamic_cast<cuda_device_queue&>(queue));
}

void* cuda_buffer_implementation::get_host_ptr_impl() const noexcept
{
    auto *block = m_allocation.get_memory_block();
    if (block && is_host_accessible(m_kind))
    {
        return block->get_data_ptr();
    }
    else
    {
        return nullptr;
    }
}

void* cuda_buffer_implementation::get_device_ptr_impl() const noexcept
{
    auto *block = m_allocation.get_memory_block();
    if (block && is_device_accessible(m_kind))
    {
        return block->get_data_ptr();
    }
    else
    {
        return nullptr;
    }
}

} // namespace hardware
} // namespace xmipp4
