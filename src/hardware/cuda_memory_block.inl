// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_block.hpp"

#include <xmipp4/core/memory/align.hpp>

namespace xmipp4
{
namespace hardware
{

inline
cuda_memory_block::cuda_memory_block(
    void *base_ptr, 
    std::ptrdiff_t offset,
    std::size_t size, 
    const cuda_device_queue *queue
) noexcept
    : m_queue(queue)
    , m_size(size)
    , m_base_ptr(base_ptr)
    , m_offset(offset)
{
}

inline
void* cuda_memory_block::get_base_ptr() const noexcept
{
    return m_base_ptr;
}

inline
std::ptrdiff_t cuda_memory_block::get_offset() const noexcept
{
    return m_offset;
}

inline
void* cuda_memory_block::get_data_ptr() const noexcept
{
    return static_cast<memory::byte*>(m_base_ptr) + m_offset;
}

inline
std::size_t cuda_memory_block::get_size() const noexcept
{
    return m_size;
}

inline
const cuda_device_queue* cuda_memory_block::get_queue() const noexcept
{
    return m_queue;
}



inline
bool cuda_memory_block_less::operator()(
    const cuda_memory_block &lhs, 
    const cuda_memory_block &rhs 
) const noexcept
{
    return as_tuple(lhs) < as_tuple(rhs);
}

inline
cuda_memory_block_less::tuple_type 
cuda_memory_block_less::as_tuple(const cuda_memory_block &block) noexcept
{
    return tuple_type(
        block.get_queue(),
        block.get_size(),
        block.get_data_ptr()
    );
}

} // namespace hardware
} // namespace xmipp4
