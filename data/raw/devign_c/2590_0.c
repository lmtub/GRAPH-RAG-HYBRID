int get_physical_address (CPUState *env, mmu_ctx_t *ctx, target_ulong eaddr,

                          int rw, int access_type)

{

    int ret;



#if 0

    qemu_log("%s\n", __func__);

#endif

    if ((access_type == ACCESS_CODE && msr_ir == 0) ||

        (access_type != ACCESS_CODE && msr_dr == 0)) {

        /* No address translation */

        ret = check_physical(env, ctx, eaddr, rw);

    } else {

        ret = -1;

        switch (env->mmu_model) {

        case POWERPC_MMU_32B:

        case POWERPC_MMU_601:

        case POWERPC_MMU_SOFT_6xx:

        case POWERPC_MMU_SOFT_74xx:

#if defined(TARGET_PPC64)

        case POWERPC_MMU_620:

        case POWERPC_MMU_64B:

#endif

            /* Try to find a BAT */

            if (env->nb_BATs != 0)

                ret = get_bat(env, ctx, eaddr, rw, access_type);

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

        case POWERPC_MMU_BOOKE:

            ret = mmubooke_get_physical_address(env, ctx, eaddr,

                                                rw, access_type);

            break;

        case POWERPC_MMU_MPC8xx:

            /* XXX: TODO */

            cpu_abort(env, "MPC8xx MMU model is not implemented\n");

            break;

        case POWERPC_MMU_BOOKE_FSL:

            /* XXX: TODO */

            cpu_abort(env, "BookE FSL MMU model not implemented\n");

            return -1;

        case POWERPC_MMU_REAL:

            cpu_abort(env, "PowerPC in real mode do not do any translation\n");

            return -1;

        default:

            cpu_abort(env, "Unknown or invalid MMU model\n");

            return -1;

        }

    }

#if 0

    qemu_log("%s address " ADDRX " => %d " PADDRX "\n",

                __func__, eaddr, ret, ctx->raddr);

#endif



    return ret;

}
