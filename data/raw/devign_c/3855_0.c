static inline int get_bat(CPUPPCState *env, mmu_ctx_t *ctx,

                          target_ulong virtual, int rw, int type)

{

    target_ulong *BATlt, *BATut, *BATu, *BATl;

    target_ulong BEPIl, BEPIu, bl;

    int i, valid, prot;

    int ret = -1;



    LOG_BATS("%s: %cBAT v " TARGET_FMT_lx "\n", __func__,

             type == ACCESS_CODE ? 'I' : 'D', virtual);

    switch (type) {

    case ACCESS_CODE:

        BATlt = env->IBAT[1];

        BATut = env->IBAT[0];

        break;

    default:

        BATlt = env->DBAT[1];

        BATut = env->DBAT[0];

        break;

    }

    for (i = 0; i < env->nb_BATs; i++) {

        BATu = &BATut[i];

        BATl = &BATlt[i];

        BEPIu = *BATu & 0xF0000000;

        BEPIl = *BATu & 0x0FFE0000;

        if (unlikely(env->mmu_model == POWERPC_MMU_601)) {

            bat_601_size_prot(env, &bl, &valid, &prot, BATu, BATl);

        } else {

            bat_size_prot(env, &bl, &valid, &prot, BATu, BATl);

        }

        LOG_BATS("%s: %cBAT%d v " TARGET_FMT_lx " BATu " TARGET_FMT_lx

                 " BATl " TARGET_FMT_lx "\n", __func__,

                 type == ACCESS_CODE ? 'I' : 'D', i, virtual, *BATu, *BATl);

        if ((virtual & 0xF0000000) == BEPIu &&

            ((virtual & 0x0FFE0000) & ~bl) == BEPIl) {

            /* BAT matches */

            if (valid != 0) {

                /* Get physical address */

                ctx->raddr = (*BATl & 0xF0000000) |

                    ((virtual & 0x0FFE0000 & bl) | (*BATl & 0x0FFE0000)) |

                    (virtual & 0x0001F000);

                /* Compute access rights */

                ctx->prot = prot;

                ret = check_prot(ctx->prot, rw, type);

                if (ret == 0) {

                    LOG_BATS("BAT %d match: r " TARGET_FMT_plx " prot=%c%c\n",

                             i, ctx->raddr, ctx->prot & PAGE_READ ? 'R' : '-',

                             ctx->prot & PAGE_WRITE ? 'W' : '-');

                }

                break;

            }

        }

    }

    if (ret < 0) {

#if defined(DEBUG_BATS)

        if (qemu_log_enabled()) {

            LOG_BATS("no BAT match for " TARGET_FMT_lx ":\n", virtual);

            for (i = 0; i < 4; i++) {

                BATu = &BATut[i];

                BATl = &BATlt[i];

                BEPIu = *BATu & 0xF0000000;

                BEPIl = *BATu & 0x0FFE0000;

                bl = (*BATu & 0x00001FFC) << 15;

                LOG_BATS("%s: %cBAT%d v " TARGET_FMT_lx " BATu " TARGET_FMT_lx

                         " BATl " TARGET_FMT_lx "\n\t" TARGET_FMT_lx " "

                         TARGET_FMT_lx " " TARGET_FMT_lx "\n",

                         __func__, type == ACCESS_CODE ? 'I' : 'D', i, virtual,

                         *BATu, *BATl, BEPIu, BEPIl, bl);

            }

        }

#endif

    }

    /* No hit */

    return ret;

}
