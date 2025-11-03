// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_resource.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_pinned_memory_resource final
    : public memory_resource
{
public:
    device* get_target_device() const noexcept override;

    memory_resource_kind get_kind() const noexcept override;

    std::shared_ptr<memory_allocator> create_allocator() override;

    static cuda_pinned_memory_resource& get() noexcept;

private:    
    static cuda_pinned_memory_resource m_instance;

}; 

} // namespace hardware
} // namespace xmipp4

