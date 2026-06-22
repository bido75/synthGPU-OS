#include <assert.h>
#include <stdio.h>

#include "synthgpu_cuda.h"

typedef struct {
    size_t total_mb;
    size_t available_mb;
    const char *override_mb;
    size_t expected_mb;
} BudgetCase;

int main(void) {
    const BudgetCase cases[] = {
        {8192, 4096, NULL, 128},
        {8192, 300, NULL, 64},
        {8192, 4096, "1024", 256},
        {16384, 12000, NULL, 6528},
        {32768, 20000, NULL, 13056},
        {32768, 20000, "50000", 16000},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        size_t actual = synthgpu_vram_budget_mb(
            cases[i].total_mb, cases[i].available_mb, cases[i].override_mb);
        printf("total=%zu available=%zu override=%s expected=%zu actual=%zu\n",
               cases[i].total_mb, cases[i].available_mb,
               cases[i].override_mb ? cases[i].override_mb : "none",
               cases[i].expected_mb, actual);
        assert(actual == cases[i].expected_mb);
    }

    puts("C/Python VRAM budget parity passed");
    return 0;
}
