// SPDX-License-Identifier: GPL-3.0-only

#include <xmipp4/cuda/hardware/cuda_buffer.hpp>

#include <xmipp4/cuda/hardware/cuda_device_queue.hpp>

namespace xmipp4
{
namespace hardware
{

void cuda_buffer::record_queue(device_queue &queue, bool exclusive)
{
    return record_queue(
        dynamic_cast<cuda_device_queue&>(queue), 
        exclusive
    );
}

} // namespace hardware
} // namespace xmipp4
