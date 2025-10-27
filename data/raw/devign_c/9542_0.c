void helper_ltr_T0(void)

{

    int selector;

    SegmentCache *dt;

    uint32_t e1, e2;

    int index, type, entry_limit;

    target_ulong ptr;

    

    selector = T0 & 0xffff;

    if ((selector & 0xfffc) == 0) {

        /* NULL selector case: invalid TR */

        env->tr.base = 0;

        env->tr.limit = 0;

        env->tr.flags = 0;

    } else {

        if (selector & 0x4)

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        dt = &env->gdt;

        index = selector & ~7;

#ifdef TARGET_X86_64

        if (env->hflags & HF_LMA_MASK)

            entry_limit = 15;

        else

#endif            

            entry_limit = 7;

        if ((index + entry_limit) > dt->limit)

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        ptr = dt->base + index;

        e1 = ldl_kernel(ptr);

        e2 = ldl_kernel(ptr + 4);

        type = (e2 >> DESC_TYPE_SHIFT) & 0xf;

        if ((e2 & DESC_S_MASK) || 

            (type != 1 && type != 9))

            raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG, selector & 0xfffc);

#ifdef TARGET_X86_64

        if (env->hflags & HF_LMA_MASK) {

            uint32_t e3;

            e3 = ldl_kernel(ptr + 8);

            load_seg_cache_raw_dt(&env->tr, e1, e2);

            env->tr.base |= (target_ulong)e3 << 32;

        } else 

#endif

        {

            load_seg_cache_raw_dt(&env->tr, e1, e2);

        }

        e2 |= DESC_TSS_BUSY_MASK;

        stl_kernel(ptr + 4, e2);

    }

    env->tr.selector = selector;

}
