MSA_ST_DF(DF_BYTE,   b, helper_ret_stb_mmu, oi, GETRA())

MSA_ST_DF(DF_HALF,   h, helper_ret_stw_mmu, oi, GETRA())

MSA_ST_DF(DF_WORD,   w, helper_ret_stl_mmu, oi, GETRA())

MSA_ST_DF(DF_DOUBLE, d, helper_ret_stq_mmu, oi, GETRA())

#else

MSA_ST_DF(DF_BYTE,   b, cpu_stb_data)

MSA_ST_DF(DF_HALF,   h, cpu_stw_data)

MSA_ST_DF(DF_WORD,   w, cpu_stl_data)

MSA_ST_DF(DF_DOUBLE, d, cpu_stq_data)

#endif



void helper_cache(CPUMIPSState *env, target_ulong addr, uint32_t op)

{

#ifndef CONFIG_USER_ONLY

    target_ulong index = addr & 0x1fffffff;

    if (op == 9) {

        /* Index Store Tag */

        memory_region_dispatch_write(env->itc_tag, index, env->CP0_TagLo,

                                     8, MEMTXATTRS_UNSPECIFIED);

    } else if (op == 5) {

        /* Index Load Tag */

        memory_region_dispatch_read(env->itc_tag, index, &env->CP0_TagLo,

                                    8, MEMTXATTRS_UNSPECIFIED);

    }

#endif

}
