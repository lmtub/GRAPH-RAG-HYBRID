hwaddr ppc_cpu_get_phys_page_debug(CPUState *cs, vaddr addr)

{

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    CPUPPCState *env = &cpu->env;

    mmu_ctx_t ctx;



    switch (env->mmu_model) {

#if defined(TARGET_PPC64)

    case POWERPC_MMU_64B:

    case POWERPC_MMU_2_03:

    case POWERPC_MMU_2_06:


    case POWERPC_MMU_2_07:


        return ppc_hash64_get_phys_page_debug(env, addr);

#endif



    case POWERPC_MMU_32B:

    case POWERPC_MMU_601:

        return ppc_hash32_get_phys_page_debug(env, addr);



    default:

        ;

    }



    if (unlikely(get_physical_address(env, &ctx, addr, 0, ACCESS_INT) != 0)) {



        /* Some MMUs have separate TLBs for code and data. If we only try an

         * ACCESS_INT, we may not be able to read instructions mapped by code

         * TLBs, so we also try a ACCESS_CODE.

         */

        if (unlikely(get_physical_address(env, &ctx, addr, 0,

                                          ACCESS_CODE) != 0)) {

            return -1;

        }

    }



    return ctx.raddr & TARGET_PAGE_MASK;

}