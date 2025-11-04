// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/hardware/buffer.hpp>

namespace xmipp4 
{
namespace hardware
{

class cuda_device_queue;

class cuda_buffer
    : public buffer
{
public:
    virtual
    void* get_device_ptr() noexcept = 0;

    virtual
    const void* get_device_ptr() const noexcept = 0;

}; 

} // namespace hardware
} // namespace xmipp4
