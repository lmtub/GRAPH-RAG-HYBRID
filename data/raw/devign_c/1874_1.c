static inline void decode(DisasContext *dc, uint32_t ir)

{

    if (unlikely(qemu_loglevel_mask(CPU_LOG_TB_OP | CPU_LOG_TB_OP_OPT))) {

        tcg_gen_debug_insn_start(dc->pc);

    }



    dc->ir = ir;

    LOG_DIS("%8.8x\t", dc->ir);



    /* try guessing 'empty' instruction memory, although it may be a valid

     * instruction sequence (eg. srui r0, r0, 0) */

    if (dc->ir) {

        dc->nr_nops = 0;

    } else {

        LOG_DIS("nr_nops=%d\t", dc->nr_nops);

        dc->nr_nops++;

        if (dc->nr_nops > 4) {

            cpu_abort(dc->env, "fetching nop sequence\n");

        }

    }



    dc->opcode = EXTRACT_FIELD(ir, 26, 31);



    dc->imm5 = EXTRACT_FIELD(ir, 0, 4);

    dc->imm16 = EXTRACT_FIELD(ir, 0, 15);

    dc->imm26 = EXTRACT_FIELD(ir, 0, 25);



    dc->csr = EXTRACT_FIELD(ir, 21, 25);

    dc->r0 = EXTRACT_FIELD(ir, 21, 25);

    dc->r1 = EXTRACT_FIELD(ir, 16, 20);

    dc->r2 = EXTRACT_FIELD(ir, 11, 15);



    /* bit 31 seems to indicate insn type.  */

    if (ir & (1 << 31)) {

        dc->format = OP_FMT_RR;

    } else {

        dc->format = OP_FMT_RI;

    }



    assert(ARRAY_SIZE(decinfo) == 64);

    assert(dc->opcode < 64);



    decinfo[dc->opcode](dc);

}
