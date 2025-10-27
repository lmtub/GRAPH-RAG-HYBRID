static void tcg_out_qemu_st(TCGContext *s, const TCGArg *args, bool is64)

{

    TCGReg datalo, datahi, addrlo;

    TCGReg addrhi __attribute__((unused));

    TCGMemOpIdx oi;

    TCGMemOp opc;

#if defined(CONFIG_SOFTMMU)

    int mem_index;

    TCGMemOp s_bits;

    tcg_insn_unit *label_ptr[2];

#endif



    datalo = *args++;

    datahi = (TCG_TARGET_REG_BITS == 32 && is64 ? *args++ : 0);

    addrlo = *args++;

    addrhi = (TARGET_LONG_BITS > TCG_TARGET_REG_BITS ? *args++ : 0);

    oi = *args++;

    opc = get_memop(oi);



#if defined(CONFIG_SOFTMMU)

    mem_index = get_mmuidx(oi);

    s_bits = opc & MO_SIZE;



    tcg_out_tlb_load(s, addrlo, addrhi, mem_index, s_bits,

                     label_ptr, offsetof(CPUTLBEntry, addr_write));



    /* TLB Hit.  */

    tcg_out_qemu_st_direct(s, datalo, datahi, TCG_REG_L1, 0, 0, opc);



    /* Record the current context of a store into ldst label */

    add_qemu_ldst_label(s, false, oi, datalo, datahi, addrlo, addrhi,

                        s->code_ptr, label_ptr);

#else

    {

        int32_t offset = GUEST_BASE;

        TCGReg base = addrlo;

        int seg = 0;



        /* See comment in tcg_out_qemu_ld re zero-extension of addrlo.  */

        if (GUEST_BASE == 0 || guest_base_flags) {

            seg = guest_base_flags;

            offset = 0;

            if (TCG_TARGET_REG_BITS > TARGET_LONG_BITS) {

                seg |= P_ADDR32;

            }

        } else if (TCG_TARGET_REG_BITS == 64) {

            /* ??? Note that we can't use the same SIB addressing scheme

               as for loads, since we require L0 free for bswap.  */

            if (offset != GUEST_BASE) {

                if (TARGET_LONG_BITS == 32) {

                    tcg_out_ext32u(s, TCG_REG_L0, base);

                    base = TCG_REG_L0;

                }

                tcg_out_movi(s, TCG_TYPE_I64, TCG_REG_L1, GUEST_BASE);

                tgen_arithr(s, ARITH_ADD + P_REXW, TCG_REG_L1, base);

                base = TCG_REG_L1;

                offset = 0;

            } else if (TARGET_LONG_BITS == 32) {

                tcg_out_ext32u(s, TCG_REG_L1, base);

                base = TCG_REG_L1;

            }

        }



        tcg_out_qemu_st_direct(s, datalo, datahi, base, offset, seg, opc);

    }

#endif

}
