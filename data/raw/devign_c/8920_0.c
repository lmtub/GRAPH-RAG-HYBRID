int cpu_mb_handle_mmu_fault (CPUState *env, target_ulong address, int rw,

                               int mmu_idx, int is_softmmu)

{

    unsigned int hit;

    unsigned int mmu_available;

    int r = 1;

    int prot;



    mmu_available = 0;

    if (env->pvr.regs[0] & PVR0_USE_MMU) {

        mmu_available = 1;

        if ((env->pvr.regs[0] & PVR0_PVR_FULL_MASK)

            && (env->pvr.regs[11] & PVR11_USE_MMU) != PVR11_USE_MMU) {

            mmu_available = 0;

        }

    }



    /* Translate if the MMU is available and enabled.  */

    if (mmu_available && (env->sregs[SR_MSR] & MSR_VM)) {

        target_ulong vaddr, paddr;

        struct microblaze_mmu_lookup lu;



        hit = mmu_translate(&env->mmu, &lu, address, rw, mmu_idx);

        if (hit) {

            vaddr = address & TARGET_PAGE_MASK;

            paddr = lu.paddr + vaddr - lu.vaddr;



            DMMU(qemu_log("MMU map mmu=%d v=%x p=%x prot=%x\n",

                     mmu_idx, vaddr, paddr, lu.prot));

            r = tlb_set_page(env, vaddr,

                             paddr, lu.prot, mmu_idx, is_softmmu);

        } else {

            env->sregs[SR_EAR] = address;

            DMMU(qemu_log("mmu=%d miss v=%x\n", mmu_idx, address));



            switch (lu.err) {

                case ERR_PROT:

                    env->sregs[SR_ESR] = rw == 2 ? 17 : 16;

                    env->sregs[SR_ESR] |= (rw == 1) << 10;

                    break;

                case ERR_MISS:

                    env->sregs[SR_ESR] = rw == 2 ? 19 : 18;

                    env->sregs[SR_ESR] |= (rw == 1) << 10;

                    break;

                default:

                    abort();

                    break;

            }



            if (env->exception_index == EXCP_MMU) {

                cpu_abort(env, "recursive faults\n");

            }



            /* TLB miss.  */

            env->exception_index = EXCP_MMU;

        }

    } else {

        /* MMU disabled or not available.  */

        address &= TARGET_PAGE_MASK;

        prot = PAGE_BITS;

        r = tlb_set_page(env, address, address, prot, mmu_idx, is_softmmu);

    }

    return r;

}
