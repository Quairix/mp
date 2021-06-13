#define OFFSET 256
#define K 8

kernel void inclusive_prefix_sum(global const float* array, global float* result, uint n) {
    const uint id = get_local_id(0);

    local float evenC[OFFSET];
    local float unevenC[OFFSET];

    local float* result_source = evenC;
    if (K & 1) {
        result_source = unevenC;
    }

    const uint MIN_OFFSET = OFFSET - 1;
    const uint chunks = n / OFFSET;
    float sum = 0.0f;
    for (uint chunk = 0; chunk <= chunks; chunk++) {
        uint curr = chunk * OFFSET + id;
        evenC[id] = array[curr];

        barrier(CLK_LOCAL_MEM_FENCE);

        uint max_id = 1;
        for (uint i = 0; i <= K; i++) {
            local float* source = evenC;
            local float* dest = unevenC;

            if (i & 1) {
                local float* temp = source;
                source = dest;
                dest = temp;
            }
            if (id < max_id) {
                dest[id] = source[id];
            }
            else {
                dest[id] = source[id] + source[id - max_id];
            }
            max_id <<= 1;

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        result[curr] = sum;
        result[curr] += result_source[id];
        sum += result_source[MIN_OFFSET];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}