static void gen_st (DisasContext *ctx, uint32_t opc, int rt,

                    int base, int16_t offset)

{

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();

    int mem_idx = ctx->mem_idx;



    gen_base_offset_addr(ctx, t0, base, offset);

    gen_load_gpr(t1, rt);

    switch (opc) {

#if defined(TARGET_MIPS64)

    case OPC_SD:

        tcg_gen_qemu_st_tl(t1, t0, mem_idx, MO_TEQ |

                           ctx->default_tcg_memop_mask);

        break;

    case OPC_SDL:

        gen_helper_0e2i(sdl, t1, t0, mem_idx);

        break;

    case OPC_SDR:

        gen_helper_0e2i(sdr, t1, t0, mem_idx);

        break;

#endif




    case OPC_SW:

        tcg_gen_qemu_st_tl(t1, t0, mem_idx, MO_TEUL |

                           ctx->default_tcg_memop_mask);

        break;




    case OPC_SH:

        tcg_gen_qemu_st_tl(t1, t0, mem_idx, MO_TEUW |

                           ctx->default_tcg_memop_mask);

        break;




    case OPC_SB:

        tcg_gen_qemu_st_tl(t1, t0, mem_idx, MO_8);

        break;




    case OPC_SWL:

        gen_helper_0e2i(swl, t1, t0, mem_idx);

        break;




    case OPC_SWR:

        gen_helper_0e2i(swr, t1, t0, mem_idx);

        break;

    }

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}