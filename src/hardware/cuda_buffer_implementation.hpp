// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/cuda/hardware/cuda_buffer.hpp>

#include <xmipp4/cuda/hardware/cuda_memory_resource.hpp>

#include "cuda_memory_block_allocation.hpp"

namespace xmipp4 
{
namespace hardware
{


class cuda_buffer_implementation final
    : public cuda_buffer
{
public:
    /**
     * @brief Construct a new cuda_buffer_implementation.
     * 
     * @param size Minimum size requirement for the memory block.
     * @param alignment Minimum alignment requirement for the memory block.
     * @param queue Queue where the allocation must be available.
     * @param allocator Block allocator from which the memory is obtained.
     */
    cuda_buffer_implementation(
        std::size_t size, 
        std::size_t alignment, 
        cuda_device_queue *queue, 
        cuda_memory_block_allocator &allocator
    );
    cuda_buffer_implementation(
        const cuda_buffer_implementation &other
    ) = default;
    cuda_buffer_implementation(
        cuda_buffer_implementation &&other
    ) = default;
    ~cuda_buffer_implementation() override = default;


    cuda_buffer_implementation& operator=(
        const cuda_buffer_implementation &other
    ) = default;
    cuda_buffer_implementation& operator=(
        cuda_buffer_implementation &&other
    ) = default;

    void* get_device_ptr() noexcept override;

    const void* get_device_ptr() const noexcept override;

    void* get_host_ptr() noexcept override;

    const void* get_host_ptr() const noexcept override;

    std::size_t get_size() const noexcept override;

    cuda_memory_resource& get_memory_resource() const noexcept override;

    void record_queue(device_queue &queue, bool exclusive=false) override;

private:
    cuda_memory_block_allocation m_allocation;
    memory_resource_kind m_kind; ///< Cached memory kind for rapid evaluation.

    void* get_host_ptr_impl() const noexcept;
    void* get_device_ptr_impl() const noexcept;

}; 

} // namespace hardware
} // namespace xmipp4
