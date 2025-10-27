static void ats_write(CPUARMState *env, const ARMCPRegInfo *ri, uint64_t value)

{

    hwaddr phys_addr;

    target_ulong page_size;

    int prot;

    int ret, is_user = ri->opc2 & 2;

    int access_type = ri->opc2 & 1;



    ret = get_phys_addr(env, value, access_type, is_user,

                        &phys_addr, &prot, &page_size);

    if (extended_addresses_enabled(env)) {

        /* ret is a DFSR/IFSR value for the long descriptor

         * translation table format, but with WnR always clear.

         * Convert it to a 64-bit PAR.

         */

        uint64_t par64 = (1 << 11); /* LPAE bit always set */

        if (ret == 0) {

            par64 |= phys_addr & ~0xfffULL;

            /* We don't set the ATTR or SH fields in the PAR. */

        } else {

            par64 |= 1; /* F */

            par64 |= (ret & 0x3f) << 1; /* FS */

            /* Note that S2WLK and FSTAGE are always zero, because we don't

             * implement virtualization and therefore there can't be a stage 2

             * fault.

             */

        }

        env->cp15.par_el1 = par64;

    } else {

        /* ret is a DFSR/IFSR value for the short descriptor

         * translation table format (with WnR always clear).

         * Convert it to a 32-bit PAR.

         */

        if (ret == 0) {

            /* We do not set any attribute bits in the PAR */

            if (page_size == (1 << 24)

                && arm_feature(env, ARM_FEATURE_V7)) {

                env->cp15.par_el1 = (phys_addr & 0xff000000) | 1 << 1;

            } else {

                env->cp15.par_el1 = phys_addr & 0xfffff000;

            }

        } else {

            env->cp15.par_el1 = ((ret & (1 << 10)) >> 5) |

                ((ret & (1 << 12)) >> 6) |

                ((ret & 0xf) << 1) | 1;

        }

    }

}
