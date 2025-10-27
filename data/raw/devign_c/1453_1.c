void ppc_tlb_invalidate_one (CPUPPCState *env, target_ulong addr)

{

#if !defined(FLUSH_ALL_TLBS)

    addr &= TARGET_PAGE_MASK;

    switch (env->mmu_model) {

    case POWERPC_MMU_SOFT_6xx:

    case POWERPC_MMU_SOFT_74xx:

        ppc6xx_tlb_invalidate_virt(env, addr, 0);

        if (env->id_tlbs == 1)

            ppc6xx_tlb_invalidate_virt(env, addr, 1);

        break;

    case POWERPC_MMU_SOFT_4xx:

    case POWERPC_MMU_SOFT_4xx_Z:

        ppc4xx_tlb_invalidate_virt(env, addr, env->spr[SPR_40x_PID]);

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

    case POWERPC_MMU_32B:

    case POWERPC_MMU_601:

        /* tlbie invalidate TLBs for all segments */

        addr &= ~((target_ulong)-1 << 28);

        /* XXX: this case should be optimized,

         * giving a mask to tlb_flush_page

         */

        tlb_flush_page(env, addr | (0x0 << 28));

        tlb_flush_page(env, addr | (0x1 << 28));

        tlb_flush_page(env, addr | (0x2 << 28));

        tlb_flush_page(env, addr | (0x3 << 28));

        tlb_flush_page(env, addr | (0x4 << 28));

        tlb_flush_page(env, addr | (0x5 << 28));

        tlb_flush_page(env, addr | (0x6 << 28));

        tlb_flush_page(env, addr | (0x7 << 28));

        tlb_flush_page(env, addr | (0x8 << 28));

        tlb_flush_page(env, addr | (0x9 << 28));

        tlb_flush_page(env, addr | (0xA << 28));

        tlb_flush_page(env, addr | (0xB << 28));

        tlb_flush_page(env, addr | (0xC << 28));

        tlb_flush_page(env, addr | (0xD << 28));

        tlb_flush_page(env, addr | (0xE << 28));

        tlb_flush_page(env, addr | (0xF << 28));

        break;

#if defined(TARGET_PPC64)

    case POWERPC_MMU_64B:

        /* tlbie invalidate TLBs for all segments */

        /* XXX: given the fact that there are too many segments to invalidate,

         *      and we still don't have a tlb_flush_mask(env, n, mask) in Qemu,

         *      we just invalidate all TLBs

         */

        tlb_flush(env, 1);

        break;

#endif /* defined(TARGET_PPC64) */

    default:

        /* XXX: TODO */

        cpu_abort(env, "Unknown MMU model\n");

        break;

    }

#else

    ppc_tlb_invalidate_all(env);

#endif

}
