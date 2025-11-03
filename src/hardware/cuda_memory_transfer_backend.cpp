// SPDX-License-Identifier: GPL-3.0-only

#include "cuda_memory_transfer_backend.hpp"

#include <xmipp4/core/hardware/memory_transfer_manager.hpp>

namespace xmipp4
{
namespace hardware
{

std::shared_ptr<memory_transfer> 
cuda_memory_transfer_backend::create_transfer(
    const memory_resource& src,
    const memory_resource& dst
) const
{
    // TODO
}

bool cuda_memory_transfer_backend::register_at(memory_transfer_manager &manager)
{
    return manager.register_backend(
        std::make_unique<cuda_memory_transfer_backend>()
    );
}

} // namespace hardware
} // namespace xmipp4
