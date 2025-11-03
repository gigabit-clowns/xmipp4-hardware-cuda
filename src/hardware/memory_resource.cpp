// SPDX-License-Identifier: GPL-3.0-only

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

#include "cuda_memory_allocator.hpp"

namespace xmipp4
{
namespace hardware
{

std::shared_ptr<cuda_memory_allocator> 
cuda_memory_resource::create_cuda_allocator()
{
    return std::make_shared<cuda_memory_allocator>(*this);
}

std::shared_ptr<memory_allocator> 
cuda_memory_resource::create_allocator()
{
    return create_allocator();
}

} // namespace hardware
} // namespace xmipp4
