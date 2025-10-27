static int get_physical_address (CPUMIPSState *env, hwaddr *physical,

                                int *prot, target_ulong real_address,

                                int rw, int access_type, int mmu_idx)

{

    /* User mode can only access useg/xuseg */

    int user_mode = mmu_idx == MIPS_HFLAG_UM;

    int supervisor_mode = mmu_idx == MIPS_HFLAG_SM;

    int kernel_mode = !user_mode && !supervisor_mode;

#if defined(TARGET_MIPS64)

    int UX = (env->CP0_Status & (1 << CP0St_UX)) != 0;

    int SX = (env->CP0_Status & (1 << CP0St_SX)) != 0;

    int KX = (env->CP0_Status & (1 << CP0St_KX)) != 0;

#endif

    int ret = TLBRET_MATCH;

    /* effective address (modified for KVM T&E kernel segments) */

    target_ulong address = real_address;



#define USEG_LIMIT      0x7FFFFFFFUL

#define KSEG0_BASE      0x80000000UL

#define KSEG1_BASE      0xA0000000UL

#define KSEG2_BASE      0xC0000000UL

#define KSEG3_BASE      0xE0000000UL



#define KVM_KSEG0_BASE  0x40000000UL

#define KVM_KSEG2_BASE  0x60000000UL



    if (kvm_enabled()) {

        /* KVM T&E adds guest kernel segments in useg */

        if (real_address >= KVM_KSEG0_BASE) {

            if (real_address < KVM_KSEG2_BASE) {

                /* kseg0 */

                address += KSEG0_BASE - KVM_KSEG0_BASE;

            } else if (real_address <= USEG_LIMIT) {

                /* kseg2/3 */

                address += KSEG2_BASE - KVM_KSEG2_BASE;

            }

        }

    }



    if (address <= USEG_LIMIT) {

        /* useg */

        if (env->CP0_Status & (1 << CP0St_ERL)) {

            *physical = address & 0xFFFFFFFF;

            *prot = PAGE_READ | PAGE_WRITE;

        } else {

            ret = env->tlb->map_address(env, physical, prot, real_address, rw, access_type);

        }

#if defined(TARGET_MIPS64)

    } else if (address < 0x4000000000000000ULL) {

        /* xuseg */

        if (UX && address <= (0x3FFFFFFFFFFFFFFFULL & env->SEGMask)) {

            ret = env->tlb->map_address(env, physical, prot, real_address, rw, access_type);

        } else {

            ret = TLBRET_BADADDR;

        }

    } else if (address < 0x8000000000000000ULL) {

        /* xsseg */

        if ((supervisor_mode || kernel_mode) &&

            SX && address <= (0x7FFFFFFFFFFFFFFFULL & env->SEGMask)) {

            ret = env->tlb->map_address(env, physical, prot, real_address, rw, access_type);

        } else {

            ret = TLBRET_BADADDR;

        }

    } else if (address < 0xC000000000000000ULL) {

        /* xkphys */

        if (kernel_mode && KX &&

            (address & 0x07FFFFFFFFFFFFFFULL) <= env->PAMask) {

            *physical = address & env->PAMask;

            *prot = PAGE_READ | PAGE_WRITE;

        } else {

            ret = TLBRET_BADADDR;

        }

    } else if (address < 0xFFFFFFFF80000000ULL) {

        /* xkseg */

        if (kernel_mode && KX &&

            address <= (0xFFFFFFFF7FFFFFFFULL & env->SEGMask)) {

            ret = env->tlb->map_address(env, physical, prot, real_address, rw, access_type);

        } else {

            ret = TLBRET_BADADDR;

        }

#endif

    } else if (address < (int32_t)KSEG1_BASE) {

        /* kseg0 */

        if (kernel_mode) {

            *physical = address - (int32_t)KSEG0_BASE;

            *prot = PAGE_READ | PAGE_WRITE;

        } else {

            ret = TLBRET_BADADDR;

        }

    } else if (address < (int32_t)KSEG2_BASE) {

        /* kseg1 */

        if (kernel_mode) {

            *physical = address - (int32_t)KSEG1_BASE;

            *prot = PAGE_READ | PAGE_WRITE;

        } else {

            ret = TLBRET_BADADDR;

        }

    } else if (address < (int32_t)KSEG3_BASE) {

        /* sseg (kseg2) */

        if (supervisor_mode || kernel_mode) {

            ret = env->tlb->map_address(env, physical, prot, real_address, rw, access_type);

        } else {

            ret = TLBRET_BADADDR;

        }

    } else {

        /* kseg3 */

        /* XXX: debug segment is not emulated */

        if (kernel_mode) {

            ret = env->tlb->map_address(env, physical, prot, real_address, rw, access_type);

        } else {

            ret = TLBRET_BADADDR;

        }

    }

    return ret;

}
