int cpu_ppc_handle_mmu_fault (CPUState *env, target_ulong address, int rw,

                              int mmu_idx, int is_softmmu)

{

    mmu_ctx_t ctx;

    int access_type;

    int ret = 0;



    if (rw == 2) {

        /* code access */

        rw = 0;

        access_type = ACCESS_CODE;

    } else {

        /* data access */

        access_type = env->access_type;

    }

    ret = get_physical_address(env, &ctx, address, rw, access_type);

    if (ret == 0) {

        ret = tlb_set_page_exec(env, address & TARGET_PAGE_MASK,

                                ctx.raddr & TARGET_PAGE_MASK, ctx.prot,

                                mmu_idx, is_softmmu);

    } else if (ret < 0) {

        LOG_MMU_STATE(env);

        if (access_type == ACCESS_CODE) {

            switch (ret) {

            case -1:

                /* No matches in page tables or TLB */

                switch (env->mmu_model) {

                case POWERPC_MMU_SOFT_6xx:

                    env->exception_index = POWERPC_EXCP_IFTLB;

                    env->error_code = 1 << 18;

                    env->spr[SPR_IMISS] = address;

                    env->spr[SPR_ICMP] = 0x80000000 | ctx.ptem;

                    goto tlb_miss;

                case POWERPC_MMU_SOFT_74xx:

                    env->exception_index = POWERPC_EXCP_IFTLB;

                    goto tlb_miss_74xx;

                case POWERPC_MMU_SOFT_4xx:

                case POWERPC_MMU_SOFT_4xx_Z:

                    env->exception_index = POWERPC_EXCP_ITLB;

                    env->error_code = 0;

                    env->spr[SPR_40x_DEAR] = address;

                    env->spr[SPR_40x_ESR] = 0x00000000;

                    break;

                case POWERPC_MMU_32B:

                case POWERPC_MMU_601:

#if defined(TARGET_PPC64)

                case POWERPC_MMU_620:

                case POWERPC_MMU_64B:

#endif

                    env->exception_index = POWERPC_EXCP_ISI;

                    env->error_code = 0x40000000;

                    break;

                case POWERPC_MMU_BOOKE:

                    /* XXX: TODO */

                    cpu_abort(env, "BookE MMU model is not implemented\n");

                    return -1;

                case POWERPC_MMU_BOOKE_FSL:

                    /* XXX: TODO */

                    cpu_abort(env, "BookE FSL MMU model is not implemented\n");

                    return -1;

                case POWERPC_MMU_MPC8xx:

                    /* XXX: TODO */

                    cpu_abort(env, "MPC8xx MMU model is not implemented\n");

                    break;

                case POWERPC_MMU_REAL:

                    cpu_abort(env, "PowerPC in real mode should never raise "

                              "any MMU exceptions\n");

                    return -1;

                default:

                    cpu_abort(env, "Unknown or invalid MMU model\n");

                    return -1;

                }

                break;

            case -2:

                /* Access rights violation */

                env->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x08000000;

                break;

            case -3:

                /* No execute protection violation */

                env->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x10000000;

                break;

            case -4:

                /* Direct store exception */

                /* No code fetch is allowed in direct-store areas */

                env->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x10000000;

                break;

#if defined(TARGET_PPC64)

            case -5:

                /* No match in segment table */

                if (env->mmu_model == POWERPC_MMU_620) {

                    env->exception_index = POWERPC_EXCP_ISI;

                    /* XXX: this might be incorrect */

                    env->error_code = 0x40000000;

                } else {

                    env->exception_index = POWERPC_EXCP_ISEG;

                    env->error_code = 0;

                }

                break;

#endif

            }

        } else {

            switch (ret) {

            case -1:

                /* No matches in page tables or TLB */

                switch (env->mmu_model) {

                case POWERPC_MMU_SOFT_6xx:

                    if (rw == 1) {

                        env->exception_index = POWERPC_EXCP_DSTLB;

                        env->error_code = 1 << 16;

                    } else {

                        env->exception_index = POWERPC_EXCP_DLTLB;

                        env->error_code = 0;

                    }

                    env->spr[SPR_DMISS] = address;

                    env->spr[SPR_DCMP] = 0x80000000 | ctx.ptem;

                tlb_miss:

                    env->error_code |= ctx.key << 19;

                    env->spr[SPR_HASH1] = ctx.pg_addr[0];

                    env->spr[SPR_HASH2] = ctx.pg_addr[1];

                    break;

                case POWERPC_MMU_SOFT_74xx:

                    if (rw == 1) {

                        env->exception_index = POWERPC_EXCP_DSTLB;

                    } else {

                        env->exception_index = POWERPC_EXCP_DLTLB;

                    }

                tlb_miss_74xx:

                    /* Implement LRU algorithm */

                    env->error_code = ctx.key << 19;

                    env->spr[SPR_TLBMISS] = (address & ~((target_ulong)0x3)) |

                        ((env->last_way + 1) & (env->nb_ways - 1));

                    env->spr[SPR_PTEHI] = 0x80000000 | ctx.ptem;

                    break;

                case POWERPC_MMU_SOFT_4xx:

                case POWERPC_MMU_SOFT_4xx_Z:

                    env->exception_index = POWERPC_EXCP_DTLB;

                    env->error_code = 0;

                    env->spr[SPR_40x_DEAR] = address;

                    if (rw)

                        env->spr[SPR_40x_ESR] = 0x00800000;

                    else

                        env->spr[SPR_40x_ESR] = 0x00000000;

                    break;

                case POWERPC_MMU_32B:

                case POWERPC_MMU_601:

#if defined(TARGET_PPC64)

                case POWERPC_MMU_620:

                case POWERPC_MMU_64B:

#endif

                    env->exception_index = POWERPC_EXCP_DSI;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                    if (rw == 1)

                        env->spr[SPR_DSISR] = 0x42000000;

                    else

                        env->spr[SPR_DSISR] = 0x40000000;

                    break;

                case POWERPC_MMU_MPC8xx:

                    /* XXX: TODO */

                    cpu_abort(env, "MPC8xx MMU model is not implemented\n");

                    break;

                case POWERPC_MMU_BOOKE:

                    /* XXX: TODO */

                    cpu_abort(env, "BookE MMU model is not implemented\n");

                    return -1;

                case POWERPC_MMU_BOOKE_FSL:

                    /* XXX: TODO */

                    cpu_abort(env, "BookE FSL MMU model is not implemented\n");

                    return -1;

                case POWERPC_MMU_REAL:

                    cpu_abort(env, "PowerPC in real mode should never raise "

                              "any MMU exceptions\n");

                    return -1;

                default:

                    cpu_abort(env, "Unknown or invalid MMU model\n");

                    return -1;

                }

                break;

            case -2:

                /* Access rights violation */

                env->exception_index = POWERPC_EXCP_DSI;

                env->error_code = 0;

                env->spr[SPR_DAR] = address;

                if (rw == 1)

                    env->spr[SPR_DSISR] = 0x0A000000;

                else

                    env->spr[SPR_DSISR] = 0x08000000;

                break;

            case -4:

                /* Direct store exception */

                switch (access_type) {

                case ACCESS_FLOAT:

                    /* Floating point load/store */

                    env->exception_index = POWERPC_EXCP_ALIGN;

                    env->error_code = POWERPC_EXCP_ALIGN_FP;

                    env->spr[SPR_DAR] = address;

                    break;

                case ACCESS_RES:

                    /* lwarx, ldarx or stwcx. */

                    env->exception_index = POWERPC_EXCP_DSI;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                    if (rw == 1)

                        env->spr[SPR_DSISR] = 0x06000000;

                    else

                        env->spr[SPR_DSISR] = 0x04000000;

                    break;

                case ACCESS_EXT:

                    /* eciwx or ecowx */

                    env->exception_index = POWERPC_EXCP_DSI;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                    if (rw == 1)

                        env->spr[SPR_DSISR] = 0x06100000;

                    else

                        env->spr[SPR_DSISR] = 0x04100000;

                    break;

                default:

                    printf("DSI: invalid exception (%d)\n", ret);

                    env->exception_index = POWERPC_EXCP_PROGRAM;

                    env->error_code =

                        POWERPC_EXCP_INVAL | POWERPC_EXCP_INVAL_INVAL;

                    env->spr[SPR_DAR] = address;

                    break;

                }

                break;

#if defined(TARGET_PPC64)

            case -5:

                /* No match in segment table */

                if (env->mmu_model == POWERPC_MMU_620) {

                    env->exception_index = POWERPC_EXCP_DSI;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                    /* XXX: this might be incorrect */

                    if (rw == 1)

                        env->spr[SPR_DSISR] = 0x42000000;

                    else

                        env->spr[SPR_DSISR] = 0x40000000;

                } else {

                    env->exception_index = POWERPC_EXCP_DSEG;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                }

                break;

#endif

            }

        }

#if 0

        printf("%s: set exception to %d %02x\n", __func__,

               env->exception, env->error_code);

#endif

        ret = 1;

    }



    return ret;

}
