static void tcg_out_qemu_ld_slow_path (TCGContext *s, TCGLabelQemuLdst *label)

{

    int s_bits;

    int ir;

    int opc = label->opc;

    int mem_index = label->mem_index;

    int data_reg = label->datalo_reg;

    int data_reg2 = label->datahi_reg;

    int addr_reg = label->addrlo_reg;

    uint8_t *raddr = label->raddr;

    uint8_t **label_ptr = &label->label_ptr[0];



    s_bits = opc & 3;



    /* resolve label address */

    reloc_pc14 (label_ptr[0], (tcg_target_long) s->code_ptr);



    /* slow path */

    ir = 3;

    tcg_out_mov (s, TCG_TYPE_I32, ir++, TCG_AREG0);

#if TARGET_LONG_BITS == 32

    tcg_out_mov (s, TCG_TYPE_I32, ir++, addr_reg);

#else

#ifdef TCG_TARGET_CALL_ALIGN_ARGS

    ir |= 1;

#endif

    tcg_out_mov (s, TCG_TYPE_I32, ir++, label->addrhi_reg);

    tcg_out_mov (s, TCG_TYPE_I32, ir++, addr_reg);

#endif

    tcg_out_movi (s, TCG_TYPE_I32, ir, mem_index);

    tcg_out_call (s, (tcg_target_long) qemu_ld_helpers[s_bits], 1);

    tcg_out32 (s, B | 8);

    tcg_out32 (s, (tcg_target_long) raddr);

    switch (opc) {

    case 0|4:

        tcg_out32 (s, EXTSB | RA (data_reg) | RS (3));

        break;

    case 1|4:

        tcg_out32 (s, EXTSH | RA (data_reg) | RS (3));

        break;

    case 0:

    case 1:

    case 2:

        if (data_reg != 3)

            tcg_out_mov (s, TCG_TYPE_I32, data_reg, 3);

        break;

    case 3:

        if (data_reg == 3) {

            if (data_reg2 == 4) {

                tcg_out_mov (s, TCG_TYPE_I32, 0, 4);

                tcg_out_mov (s, TCG_TYPE_I32, 4, 3);

                tcg_out_mov (s, TCG_TYPE_I32, 3, 0);

            }

            else {

                tcg_out_mov (s, TCG_TYPE_I32, data_reg2, 3);

                tcg_out_mov (s, TCG_TYPE_I32, 3, 4);

            }

        }

        else {

            if (data_reg != 4) tcg_out_mov (s, TCG_TYPE_I32, data_reg, 4);

            if (data_reg2 != 3) tcg_out_mov (s, TCG_TYPE_I32, data_reg2, 3);

        }

        break;

    }

    /* Jump to the code corresponding to next IR of qemu_st */

    tcg_out_b (s, 0, (tcg_target_long) raddr);

}
