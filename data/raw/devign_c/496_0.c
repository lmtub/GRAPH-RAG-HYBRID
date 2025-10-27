int ppc_hash32_handle_mmu_fault(CPUPPCState *env, target_ulong address, int rw,

                                int mmu_idx)

{

    struct mmu_ctx_hash32 ctx;

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

    ret = ppc_hash32_get_physical_address(env, &ctx, address, rw, access_type);

    if (ret == 0) {

        tlb_set_page(env, address & TARGET_PAGE_MASK,

                     ctx.raddr & TARGET_PAGE_MASK, ctx.prot,

                     mmu_idx, TARGET_PAGE_SIZE);

        ret = 0;

    } else if (ret < 0) {

        LOG_MMU_STATE(env);

        if (access_type == ACCESS_CODE) {

            switch (ret) {

            case -1:

                /* No matches in page tables or TLB */

                env->exception_index = POWERPC_EXCP_ISI;

                env->error_code = 0x40000000;

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

            }

        } else {

            switch (ret) {

            case -1:

                /* No matches in page tables or TLB */

                env->exception_index = POWERPC_EXCP_DSI;

                env->error_code = 0;

                env->spr[SPR_DAR] = address;

                if (rw == 1) {

                    env->spr[SPR_DSISR] = 0x42000000;

                } else {

                    env->spr[SPR_DSISR] = 0x40000000;

                }

                break;

            case -2:

                /* Access rights violation */

                env->exception_index = POWERPC_EXCP_DSI;

                env->error_code = 0;

                env->spr[SPR_DAR] = address;

                if (rw == 1) {

                    env->spr[SPR_DSISR] = 0x0A000000;

                } else {

                    env->spr[SPR_DSISR] = 0x08000000;

                }

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

                    if (rw == 1) {

                        env->spr[SPR_DSISR] = 0x06000000;

                    } else {

                        env->spr[SPR_DSISR] = 0x04000000;

                    }

                    break;

                case ACCESS_EXT:

                    /* eciwx or ecowx */

                    env->exception_index = POWERPC_EXCP_DSI;

                    env->error_code = 0;

                    env->spr[SPR_DAR] = address;

                    if (rw == 1) {

                        env->spr[SPR_DSISR] = 0x06100000;

                    } else {

                        env->spr[SPR_DSISR] = 0x04100000;

                    }

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
