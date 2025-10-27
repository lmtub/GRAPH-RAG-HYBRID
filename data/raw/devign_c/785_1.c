int get_physical_address (CPUState *env, mmu_ctx_t *ctx, target_ulong eaddr,

                          int rw, int access_type, int check_BATs)

{

    int ret;

#if 0

    if (loglevel != 0) {

        fprintf(logfile, "%s\n", __func__);

    }

#endif

    if ((access_type == ACCESS_CODE && msr_ir == 0) ||

        (access_type != ACCESS_CODE && msr_dr == 0)) {

        /* No address translation */

        ret = check_physical(env, ctx, eaddr, rw);

    } else {

        ret = -1;

        switch (env->mmu_model) {

        case POWERPC_MMU_32B:

        case POWERPC_MMU_SOFT_6xx:

        case POWERPC_MMU_SOFT_74xx:

            /* Try to find a BAT */

            if (check_BATs)

                ret = get_bat(env, ctx, eaddr, rw, access_type);

            /* No break here */

#if defined(TARGET_PPC64)

        case POWERPC_MMU_64B:

        case POWERPC_MMU_64BRIDGE:

#endif

            if (ret < 0) {

                /* We didn't match any BAT entry or don't have BATs */

                ret = get_segment(env, ctx, eaddr, rw, access_type);

            }

            break;

        case POWERPC_MMU_SOFT_4xx:

        case POWERPC_MMU_SOFT_4xx_Z:

            ret = mmu40x_get_physical_address(env, ctx, eaddr,

                                              rw, access_type);

            break;

        case POWERPC_MMU_601:

            /* XXX: TODO */

            cpu_abort(env, "601 MMU model not implemented\n");

            return -1;

        case POWERPC_MMU_BOOKE:

            ret = mmubooke_get_physical_address(env, ctx, eaddr,

                                                rw, access_type);

            break;

        case POWERPC_MMU_BOOKE_FSL:

            /* XXX: TODO */

            cpu_abort(env, "BookE FSL MMU model not implemented\n");

            return -1;

        case POWERPC_MMU_REAL_4xx:

            cpu_abort(env, "PowerPC 401 does not do any translation\n");

            return -1;

        default:

            cpu_abort(env, "Unknown or invalid MMU model\n");

            return -1;

        }

    }

#if 0

    if (loglevel != 0) {

        fprintf(logfile, "%s address " ADDRX " => %d " PADDRX "\n",

                __func__, eaddr, ret, ctx->raddr);

    }

#endif



    return ret;

}
