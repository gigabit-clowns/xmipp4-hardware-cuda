// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_host_pinned_memory_resource final
    : public cuda_memory_resource
{
public:
    cuda_host_pinned_memory_resource(
        const cuda_host_pinned_memory_resource &other
    ) = delete; 
    cuda_host_pinned_memory_resource(
        cuda_host_pinned_memory_resource &&other
    ) = delete; 
    ~cuda_host_pinned_memory_resource() override = default;
    
    cuda_host_pinned_memory_resource& operator=(
        const cuda_host_pinned_memory_resource &other
    ) = delete;
    cuda_host_pinned_memory_resource& operator=(
        cuda_host_pinned_memory_resource &&other
    ) = delete;



    device* get_target_device() const noexcept override;

    memory_resource_kind get_kind() const noexcept override;

    std::shared_ptr<memory_allocator> create_allocator() override;

    void* malloc(std::size_t size, std::size_t alignment) override;

    void free(void* ptr) override;

    static cuda_host_pinned_memory_resource& get() noexcept;

private:    
    cuda_host_pinned_memory_resource() = default;

    static cuda_host_pinned_memory_resource m_instance;

}; 

} // namespace hardware
} // namespace xmipp4

