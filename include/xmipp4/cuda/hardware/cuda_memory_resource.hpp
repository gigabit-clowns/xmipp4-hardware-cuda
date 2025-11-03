// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/memory_resource.hpp>

#include <cstddef>

namespace xmipp4 
{
namespace hardware
{

class cuda_memory_allocator;

class cuda_memory_resource
    : public memory_resource
{
public:
    std::shared_ptr<cuda_memory_allocator> create_cuda_allocator();
    std::shared_ptr<memory_allocator> create_allocator() override;
    virtual void* malloc(std::size_t size) noexcept = 0;
    virtual void free(void* ptr) noexcept = 0;

}; 

} // namespace hardware
} // namespace xmipp4
