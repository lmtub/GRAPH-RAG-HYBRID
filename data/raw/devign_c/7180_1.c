static void tcg_out_qemu_st_slow_path(TCGContext *s, TCGLabelQemuLdst *l)

{

    TCGMemOp opc = l->opc;

    TCGMemOp s_bits = opc & MO_SIZE;

    uint8_t **label_ptr = &l->label_ptr[0];

    TCGReg retaddr;



    /* resolve label address */

    *(uint32_t *)label_ptr[0] = (uint32_t)(s->code_ptr - label_ptr[0] - 4);

    if (TARGET_LONG_BITS > TCG_TARGET_REG_BITS) {

        *(uint32_t *)label_ptr[1] = (uint32_t)(s->code_ptr - label_ptr[1] - 4);

    }



    if (TCG_TARGET_REG_BITS == 32) {

        int ofs = 0;



        tcg_out_st(s, TCG_TYPE_PTR, TCG_AREG0, TCG_REG_ESP, ofs);

        ofs += 4;



        tcg_out_st(s, TCG_TYPE_I32, l->addrlo_reg, TCG_REG_ESP, ofs);

        ofs += 4;



        if (TARGET_LONG_BITS == 64) {

            tcg_out_st(s, TCG_TYPE_I32, l->addrhi_reg, TCG_REG_ESP, ofs);

            ofs += 4;

        }



        tcg_out_st(s, TCG_TYPE_I32, l->datalo_reg, TCG_REG_ESP, ofs);

        ofs += 4;



        if (s_bits == MO_64) {

            tcg_out_st(s, TCG_TYPE_I32, l->datahi_reg, TCG_REG_ESP, ofs);

            ofs += 4;

        }



        tcg_out_sti(s, TCG_TYPE_I32, TCG_REG_ESP, ofs, l->mem_index);

        ofs += 4;



        retaddr = TCG_REG_EAX;

        tcg_out_movi(s, TCG_TYPE_I32, retaddr, (uintptr_t)l->raddr);

        tcg_out_st(s, TCG_TYPE_I32, retaddr, TCG_REG_ESP, ofs);

    } else {

        tcg_out_mov(s, TCG_TYPE_PTR, tcg_target_call_iarg_regs[0], TCG_AREG0);

        /* The second argument is already loaded with addrlo.  */

        tcg_out_mov(s, (s_bits == MO_64 ? TCG_TYPE_I64 : TCG_TYPE_I32),

                    tcg_target_call_iarg_regs[2], l->datalo_reg);

        tcg_out_movi(s, TCG_TYPE_I32, tcg_target_call_iarg_regs[3],

                     l->mem_index);



        if (ARRAY_SIZE(tcg_target_call_iarg_regs) > 4) {

            retaddr = tcg_target_call_iarg_regs[4];

            tcg_out_movi(s, TCG_TYPE_PTR, retaddr, (uintptr_t)l->raddr);

        } else {

            retaddr = TCG_REG_RAX;

            tcg_out_movi(s, TCG_TYPE_PTR, retaddr, (uintptr_t)l->raddr);

            tcg_out_st(s, TCG_TYPE_PTR, retaddr, TCG_REG_ESP, 0);

        }

    }



    /* "Tail call" to the helper, with the return address back inline.  */

    tcg_out_push(s, retaddr);

    tcg_out_jmp(s, (uintptr_t)qemu_st_helpers[opc]);

}
