static inline void validate_seg(int seg_reg, int cpl)

{

    int dpl;

    uint32_t e2;

    

    e2 = env->segs[seg_reg].flags;

    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    if (!(e2 & DESC_CS_MASK) || !(e2 & DESC_C_MASK)) {

        /* data or non conforming code segment */

        if (dpl < cpl) {

            cpu_x86_load_seg_cache(env, seg_reg, 0, 0, 0, 0);

        }

    }

}
