uint32_t HELPER(neon_acgt_f32)(uint32_t a, uint32_t b)

{

    float32 f0 = float32_abs(make_float32(a));

    float32 f1 = float32_abs(make_float32(b));

    if (float32_compare_quiet(f0, f1, NFS) == float_relation_greater) {

        return ~0;

    }

    return 0;

}
