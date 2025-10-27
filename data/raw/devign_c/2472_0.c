static void tcg_out_qemu_st (TCGContext *s, const TCGArg *args, int opc)

{

    int addr_reg, r0, r1, rbase, data_reg, mem_index, bswap;

#ifdef CONFIG_SOFTMMU

    int r2;

    void *label1_ptr, *label2_ptr;

#endif



    data_reg = *args++;

    addr_reg = *args++;

    mem_index = *args;



#ifdef CONFIG_SOFTMMU

    r0 = 3;

    r1 = 4;

    r2 = 0;

    rbase = 0;



    tcg_out_tlb_read (s, r0, r1, r2, addr_reg, opc,

                      offsetof (CPUState, tlb_table[mem_index][0].addr_write));



    tcg_out32 (s, CMP | BF (7) | RA (r2) | RB (r1) | CMP_L);



    label1_ptr = s->code_ptr;

#ifdef FAST_PATH

    tcg_out32 (s, BC | BI (7, CR_EQ) | BO_COND_TRUE);

#endif



    /* slow path */

    tcg_out_mov (s, 3, addr_reg);

    tcg_out_rld (s, RLDICL, 4, data_reg, 0, 64 - (1 << (3 + opc)));

    tcg_out_movi (s, TCG_TYPE_I64, 5, mem_index);



    tcg_out_call (s, (tcg_target_long) qemu_st_helpers[opc], 1);



    label2_ptr = s->code_ptr;

    tcg_out32 (s, B);



    /* label1: fast path */

#ifdef FAST_PATH

    reloc_pc14 (label1_ptr, (tcg_target_long) s->code_ptr);

#endif



    tcg_out32 (s, (LD_ADDEND

                   | RT (r0)

                   | RA (r0)

                   | (offsetof (CPUTLBEntry, addend)

                      - offsetof (CPUTLBEntry, addr_write))

                   ));

    /* r0 = env->tlb_table[mem_index][index].addend */

    tcg_out32 (s, ADD | RT (r0) | RA (r0) | RB (addr_reg));

    /* r0 = env->tlb_table[mem_index][index].addend + addr */



#else  /* !CONFIG_SOFTMMU */

#if TARGET_LONG_BITS == 32

    tcg_out_rld (s, RLDICL, addr_reg, addr_reg, 0, 32);

#endif

    r1 = 3;

    r0 = addr_reg;

    rbase = GUEST_BASE ? TCG_GUEST_BASE_REG : 0;

#endif



#ifdef TARGET_WORDS_BIGENDIAN

    bswap = 0;

#else

    bswap = 1;

#endif

    switch (opc) {

    case 0:

        tcg_out32 (s, STBX | SAB (data_reg, rbase, r0));

        break;

    case 1:

        if (bswap)

            tcg_out32 (s, STHBRX | SAB (data_reg, rbase, r0));

        else

            tcg_out32 (s, STHX | SAB (data_reg, rbase, r0));

        break;

    case 2:

        if (bswap)

            tcg_out32 (s, STWBRX | SAB (data_reg, rbase, r0));

        else

            tcg_out32 (s, STWX | SAB (data_reg, rbase, r0));

        break;

    case 3:

        if (bswap) {

            tcg_out32 (s, STWBRX | SAB (data_reg, rbase, r0));

            tcg_out32 (s, ADDI | RT (r1) | RA (r0) | 4);

            tcg_out_rld (s, RLDICL, 0, data_reg, 32, 0);

            tcg_out32 (s, STWBRX | SAB (0, rbase, r1));

        }

        else tcg_out32 (s, STDX | SAB (data_reg, rbase, r0));

        break;

    }



#ifdef CONFIG_SOFTMMU

    reloc_pc24 (label2_ptr, (tcg_target_long) s->code_ptr);

#endif

}
