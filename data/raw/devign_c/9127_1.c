void ppc_tlb_invalidate_all (CPUPPCState *env)

{

    switch (env->mmu_model) {

    case POWERPC_MMU_SOFT_6xx:

    case POWERPC_MMU_SOFT_74xx:

        ppc6xx_tlb_invalidate_all(env);

        break;

    case POWERPC_MMU_SOFT_4xx:

    case POWERPC_MMU_SOFT_4xx_Z:

        ppc4xx_tlb_invalidate_all(env);

        break;

    case POWERPC_MMU_REAL_4xx:

        cpu_abort(env, "No TLB for PowerPC 4xx in real mode\n");

        break;

    case POWERPC_MMU_BOOKE:

        /* XXX: TODO */

        cpu_abort(env, "MMU model not implemented\n");

        break;

    case POWERPC_MMU_BOOKE_FSL:

        /* XXX: TODO */

        cpu_abort(env, "MMU model not implemented\n");

        break;

    case POWERPC_MMU_601:

        /* XXX: TODO */

        cpu_abort(env, "MMU model not implemented\n");

        break;

    case POWERPC_MMU_32B:

#if defined(TARGET_PPC64)

    case POWERPC_MMU_64B:

    case POWERPC_MMU_64BRIDGE:

#endif /* defined(TARGET_PPC64) */

        tlb_flush(env, 1);

        break;

    default:

        /* XXX: TODO */

        cpu_abort(env, "Unknown MMU model %d\n", env->mmu_model);

        break;

    }

}
