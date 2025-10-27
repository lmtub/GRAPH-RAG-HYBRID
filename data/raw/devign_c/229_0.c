static int get_physical_address_data(CPUState *env,

                                     target_phys_addr_t *physical, int *prot,

                                     target_ulong address, int rw, int is_user)

{

    target_ulong mask;

    unsigned int i;



    if ((env->lsu & DMMU_E) == 0) { /* DMMU disabled */

        *physical = address;

        *prot = PAGE_READ | PAGE_WRITE;

        return 0;

    }



    for (i = 0; i < 64; i++) {

        switch ((env->dtlb_tte[i] >> 61) & 3) {

        default:

        case 0x0: // 8k

            mask = 0xffffffffffffe000ULL;

            break;

        case 0x1: // 64k

            mask = 0xffffffffffff0000ULL;

            break;

        case 0x2: // 512k

            mask = 0xfffffffffff80000ULL;

            break;

        case 0x3: // 4M

            mask = 0xffffffffffc00000ULL;

            break;

        }

        // ctx match, vaddr match, valid?

        if (env->dmmuregs[1] == (env->dtlb_tag[i] & 0x1fff) &&

            (address & mask) == (env->dtlb_tag[i] & mask) &&

            (env->dtlb_tte[i] & 0x8000000000000000ULL)) {

            // access ok?

            if (((env->dtlb_tte[i] & 0x4) && is_user) ||

                (!(env->dtlb_tte[i] & 0x2) && (rw == 1))) {

                if (env->dmmuregs[3]) /* Fault status register */

                    env->dmmuregs[3] = 2; /* overflow (not read before

                                             another fault) */

                env->dmmuregs[3] |= (is_user << 3) | ((rw == 1) << 2) | 1;

                env->dmmuregs[4] = address; /* Fault address register */

                env->exception_index = TT_DFAULT;

#ifdef DEBUG_MMU

                printf("DFAULT at 0x%" PRIx64 "\n", address);

#endif

                return 1;

            }

            *physical = ((env->dtlb_tte[i] & mask) | (address & ~mask)) &

                        0x1ffffffe000ULL;

            *prot = PAGE_READ;

            if (env->dtlb_tte[i] & 0x2)

                *prot |= PAGE_WRITE;

            return 0;

        }

    }

#ifdef DEBUG_MMU

    printf("DMISS at 0x%" PRIx64 "\n", address);

#endif

    env->dmmuregs[6] = (address & ~0x1fffULL) | (env->dmmuregs[1] & 0x1fff);

    env->exception_index = TT_DMISS;

    return 1;

}
