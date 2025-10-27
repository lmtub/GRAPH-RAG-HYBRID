static inline void load_seg_vm(int seg, int selector)

{

    selector &= 0xffff;

    cpu_x86_load_seg_cache(env, seg, selector, 

                           (uint8_t *)(selector << 4), 0xffff, 0);

}
