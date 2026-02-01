#include <cstdint>

extern "C" {
    // Dummy implementation of rsmi_init to satisfy linker dependencies
    // when the real ROCm SMI library is broken or causes Bus Errors (gfx1151).
    // Returns 0 (RSMI_STATUS_SUCCESS).
    __attribute__((visibility("default")))
    int rsmi_init(uint64_t flags) {
        return 0;
    }
}
