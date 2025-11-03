// SPDX-License-Identifier: GPL-3.0-only

#include <hardware/cuda_memory_block.hpp>

#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using namespace xmipp4::hardware;

TEST_CASE( "construct cuda_memory_block", "[cuda_memory_block]" )
{
    const std::uintptr_t base_ptr_value = 0xDEADBEEF;
    auto *const ptr = reinterpret_cast<void*>(ptr_value);
    const std::uintptr_t queue_value = 0xA7EBADF0D;
    auto *const queue = reinterpret_cast<cuda_device_queue*>(queue_value);

    cuda_memory_block block(base_ptr_value, 20, 0xC0FFE, queue);
    REQUIRE( block.get_base_data() == ptr );
    REQUIRE( block.get_offset() == 20 );
    REQUIRE( block.get_size() == 0xC0FFE );
    REQUIRE( block.get_queue() == queue );
}
