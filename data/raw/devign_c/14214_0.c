static int ppc_hash32_get_bat(CPUPPCState *env, struct mmu_ctx_hash32 *ctx,

                              target_ulong virtual, int rwx)

{

    target_ulong *BATlt, *BATut;

    target_ulong BEPIl, BEPIu, bl;

    int i, valid, prot;

    int ret = -1;



    LOG_BATS("%s: %cBAT v " TARGET_FMT_lx "\n", __func__,

             rwx == 2 ? 'I' : 'D', virtual);

    if (rwx == 2) {

        BATlt = env->IBAT[1];

        BATut = env->IBAT[0];

    } else {

        BATlt = env->DBAT[1];

        BATut = env->DBAT[0];

    }

    for (i = 0; i < env->nb_BATs; i++) {

        target_ulong batu = BATut[i];

        target_ulong batl = BATlt[i];



        BEPIu = batu & BATU32_BEPIU;

        BEPIl = batu & BATU32_BEPIL;

        if (unlikely(env->mmu_model == POWERPC_MMU_601)) {

            hash32_bat_601_size(env, &bl, &valid, batu, batl);

            prot = hash32_bat_601_prot(env, batu, batl);

        } else {

            hash32_bat_size(env, &bl, &valid, batu, batl);

            prot = hash32_bat_prot(env, batu, batl);

        }

        LOG_BATS("%s: %cBAT%d v " TARGET_FMT_lx " BATu " TARGET_FMT_lx

                 " BATl " TARGET_FMT_lx "\n", __func__,

                 type == ACCESS_CODE ? 'I' : 'D', i, virtual, batu, batl);

        if ((virtual & BATU32_BEPIU) == BEPIu &&

            ((virtual & BATU32_BEPIL) & ~bl) == BEPIl) {

            /* BAT matches */

            if (valid != 0) {

                /* Get physical address */

                ctx->raddr = (batl & BATL32_BRPNU) |

                    ((virtual & BATU32_BEPIL & bl) | (batl & BATL32_BRPNL)) |

                    (virtual & 0x0001F000);

                /* Compute access rights */

                ctx->prot = prot;

                ret = ppc_hash32_check_prot(ctx->prot, rwx);

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

                BEPIu = *BATu & BATU32_BEPIU;

                BEPIl = *BATu & BATU32_BEPIL;

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
