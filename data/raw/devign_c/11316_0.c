static bool get_phys_addr_pmsav8(CPUARMState *env, uint32_t address,

                                 MMUAccessType access_type, ARMMMUIdx mmu_idx,

                                 hwaddr *phys_ptr, int *prot, uint32_t *fsr)

{

    ARMCPU *cpu = arm_env_get_cpu(env);

    bool is_user = regime_is_user(env, mmu_idx);

    int n;

    int matchregion = -1;

    bool hit = false;



    *phys_ptr = address;

    *prot = 0;



    /* Unlike the ARM ARM pseudocode, we don't need to check whether this

     * was an exception vector read from the vector table (which is always

     * done using the default system address map), because those accesses

     * are done in arm_v7m_load_vector(), which always does a direct

     * read using address_space_ldl(), rather than going via this function.

     */

    if (regime_translation_disabled(env, mmu_idx)) { /* MPU disabled */

        hit = true;

    } else if (m_is_ppb_region(env, address)) {

        hit = true;

    } else if (pmsav7_use_background_region(cpu, mmu_idx, is_user)) {

        hit = true;

    } else {

        for (n = (int)cpu->pmsav7_dregion - 1; n >= 0; n--) {

            /* region search */

            /* Note that the base address is bits [31:5] from the register

             * with bits [4:0] all zeroes, but the limit address is bits

             * [31:5] from the register with bits [4:0] all ones.

             */

            uint32_t base = env->pmsav8.rbar[n] & ~0x1f;

            uint32_t limit = env->pmsav8.rlar[n] | 0x1f;



            if (!(env->pmsav8.rlar[n] & 0x1)) {

                /* Region disabled */

                continue;

            }



            if (address < base || address > limit) {

                continue;

            }



            if (hit) {

                /* Multiple regions match -- always a failure (unlike

                 * PMSAv7 where highest-numbered-region wins)

                 */

                *fsr = 0x00d; /* permission fault */

                return true;

            }



            matchregion = n;

            hit = true;



            if (base & ~TARGET_PAGE_MASK) {

                qemu_log_mask(LOG_UNIMP,

                              "MPU_RBAR[%d]: No support for MPU region base"

                              "address of 0x%" PRIx32 ". Minimum alignment is "

                              "%d\n",

                              n, base, TARGET_PAGE_BITS);

                continue;

            }

            if ((limit + 1) & ~TARGET_PAGE_MASK) {

                qemu_log_mask(LOG_UNIMP,

                              "MPU_RBAR[%d]: No support for MPU region limit"

                              "address of 0x%" PRIx32 ". Minimum alignment is "

                              "%d\n",

                              n, limit, TARGET_PAGE_BITS);

                continue;

            }

        }

    }



    if (!hit) {

        /* background fault */

        *fsr = 0;

        return true;

    }



    if (matchregion == -1) {

        /* hit using the background region */

        get_phys_addr_pmsav7_default(env, mmu_idx, address, prot);

    } else {

        uint32_t ap = extract32(env->pmsav8.rbar[matchregion], 1, 2);

        uint32_t xn = extract32(env->pmsav8.rbar[matchregion], 0, 1);



        if (m_is_system_region(env, address)) {

            /* System space is always execute never */

            xn = 1;

        }



        *prot = simple_ap_to_rw_prot(env, mmu_idx, ap);

        if (*prot && !xn) {

            *prot |= PAGE_EXEC;

        }

        /* We don't need to look the attribute up in the MAIR0/MAIR1

         * registers because that only tells us about cacheability.

         */

    }



    *fsr = 0x00d; /* Permission fault */

    return !(*prot & (1 << access_type));

}
