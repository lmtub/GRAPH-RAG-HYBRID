static void gen_mtspr(DisasContext *ctx)

{

    void (*write_cb)(DisasContext *ctx, int sprn, int gprn);

    uint32_t sprn = SPR(ctx->opcode);



#if defined(CONFIG_USER_ONLY)

    write_cb = ctx->spr_cb[sprn].uea_write;

#else

    if (ctx->pr) {

        write_cb = ctx->spr_cb[sprn].uea_write;

    } else if (ctx->hv) {

        write_cb = ctx->spr_cb[sprn].hea_write;

    } else {

        write_cb = ctx->spr_cb[sprn].oea_write;

    }

#endif

    if (likely(write_cb != NULL)) {

        if (likely(write_cb != SPR_NOACCESS)) {

            (*write_cb)(ctx, sprn, rS(ctx->opcode));

        } else {

            /* Privilege exception */

            fprintf(stderr, "Trying to write privileged spr %d (0x%03x) at "

                    TARGET_FMT_lx "\n", sprn, sprn, ctx->nip - 4);

            if (qemu_log_separate()) {

                qemu_log("Trying to write privileged spr %d (0x%03x) at "

                         TARGET_FMT_lx "\n", sprn, sprn, ctx->nip - 4);

            }

            gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        }

    } else {

        /* Not defined */

        if (qemu_log_separate()) {

            qemu_log("Trying to write invalid spr %d (0x%03x) at "

                     TARGET_FMT_lx "\n", sprn, sprn, ctx->nip - 4);

        }

        fprintf(stderr, "Trying to write invalid spr %d (0x%03x) at "

                TARGET_FMT_lx "\n", sprn, sprn, ctx->nip - 4);

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_SPR);

    }

}
