// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_pinned_memory_resource.hpp"

#include <utility>

namespace xmipp4
{
namespace hardware
{

cuda_pinned_memory_resource cuda_pinned_memory_resource::m_instance;

device* cuda_pinned_memory_resource::get_target_device() const noexcept
{
    return nullptr;
}

memory_resource_kind cuda_pinned_memory_resource::get_kind() const noexcept
{
    return memory_resource_kind::device_mapped;
}

std::shared_ptr<memory_allocator> 
cuda_pinned_memory_resource::create_allocator()
{
    return nullptr; // TODO
}

cuda_pinned_memory_resource& cuda_pinned_memory_resource::get() noexcept
{
    return m_instance;
}

} // namespace hardware
} // namespace xmipp4
