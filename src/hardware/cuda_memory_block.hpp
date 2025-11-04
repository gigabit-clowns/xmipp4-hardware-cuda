// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <xmipp4/core/span.hpp>

#include <cstddef>

namespace xmipp4 
{
namespace hardware
{

class cuda_device_queue;

/**
 * @brief Represents a chunk of data managed by cuda_memory_cache.
 * 
 * It contains an unique id to a queue where this data is synchronous.
 * It also contains the size of the referenced block and a pointer 
 * to its data.
 * 
 */
class cuda_memory_block
{
public:
    /**
     * @brief Construct a new cuda memory block from its components.
     * 
     * @param base_ptr Pointer to the super-block allocated by CUDA.
     * @param offset Offset into the super-block.
     * @param size Number of bytes referenced.
     * @param queue Queue where this belongs.
     */
    cuda_memory_block(
        void *data_ptr,
        std::size_t size, 
        const cuda_device_queue *queue 
    ) noexcept;

    cuda_memory_block(const cuda_memory_block &other) = default;
    cuda_memory_block(cuda_memory_block &&other) = default;
    ~cuda_memory_block() = default;

    cuda_memory_block& operator=(const cuda_memory_block &other) = default;
    cuda_memory_block& operator=(cuda_memory_block &&other) = default;

    /**
     * @brief Get the pointer to the data.
     * 
     * @return void* 
     */
    void* get_data_ptr() const noexcept;

    /**
     * @brief Get the number of bytes referenced by this object.
     * 
     * @return std::size_t Number of bytes.
     */
    std::size_t get_size() const noexcept;

    /**
     * @brief Get the queue where this block belongs to.
     * 
     * @return const cuda_device_queue* Pointer to the queue.
     */
    const cuda_device_queue* get_queue() const noexcept;

private:
    const cuda_device_queue *m_queue;
    std::size_t m_size;
    void *m_data_ptr;

}; 



/**
 * @brief Lexicographically compare two cuda_memory_block objects.
 * 
 * First, queue IDs are compared.
 * If equal, then, sizes are compared.
 * If equal, then alignments are compared.
 * If equal, data pointers are compared.
 * 
 */
class cuda_memory_block_less
{
public:
    bool operator()(
        const cuda_memory_block &lhs, 
        const cuda_memory_block &rhs
    ) const noexcept;

private:
    using tuple_type = std::tuple<
        const cuda_device_queue*,
        std::size_t,
        void*
    >;

    static
    tuple_type as_tuple(const cuda_memory_block &block) noexcept;

};

} // namespace hardware
} // namespace xmipp4

#include "cuda_memory_block.inl"
