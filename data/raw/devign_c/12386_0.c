static int get_physical_address_code(CPUState *env,

                                     target_phys_addr_t *physical, int *prot,

                                     target_ulong address, int mmu_idx)

{

    unsigned int i;

    uint64_t context;



    int is_user = (mmu_idx == MMU_USER_IDX ||

                   mmu_idx == MMU_USER_SECONDARY_IDX);



    if ((env->lsu & IMMU_E) == 0 || (env->pstate & PS_RED) != 0) {

        /* IMMU disabled */

        *physical = ultrasparc_truncate_physical(address);

        *prot = PAGE_EXEC;

        return 0;

    }



    if (env->tl == 0) {

        /* PRIMARY context */

        context = env->dmmu.mmu_primary_context & 0x1fff;

    } else {

        /* NUCLEUS context */

        context = 0;

    }



    for (i = 0; i < 64; i++) {

        // ctx match, vaddr match, valid?

        if (ultrasparc_tag_match(&env->itlb[i],

                                 address, context, physical)) {

            // access ok?

            if ((env->itlb[i].tte & 0x4) && is_user) {

                if (env->immu.sfsr) /* Fault status register */

                    env->immu.sfsr = 2; /* overflow (not read before

                                             another fault) */

                env->immu.sfsr |= (is_user << 3) | 1;

                env->exception_index = TT_TFAULT;

#ifdef DEBUG_MMU

                printf("TFAULT at 0x%" PRIx64 "\n", address);

#endif

                return 1;

            }

            *prot = PAGE_EXEC;

            TTE_SET_USED(env->itlb[i].tte);

            return 0;

        }

    }

#ifdef DEBUG_MMU

    printf("TMISS at 0x%" PRIx64 "\n", address);

#endif

    /* Context is stored in DMMU (dmmuregs[1]) also for IMMU */

    env->immu.tag_access = (address & ~0x1fffULL) | context;

    env->exception_index = TT_TMISS;

    return 1;

}
