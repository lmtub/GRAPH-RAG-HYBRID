static void tcg_out_qemu_ld(TCGContext* s, TCGReg data_reg, TCGReg addr_reg,

                            TCGMemOpIdx oi)

{

    TCGMemOp opc = get_memop(oi);

#ifdef CONFIG_SOFTMMU

    unsigned mem_index = get_mmuidx(oi);

    tcg_insn_unit *label_ptr;

    TCGReg base_reg;



    base_reg = tcg_out_tlb_read(s, addr_reg, opc, mem_index, 1);



    label_ptr = s->code_ptr + 1;

    tcg_out_insn(s, RI, BRC, S390_CC_NE, 0);



    tcg_out_qemu_ld_direct(s, opc, data_reg, base_reg, TCG_REG_R2, 0);



    add_qemu_ldst_label(s, 1, oi, data_reg, addr_reg, s->code_ptr, label_ptr);

#else

    TCGReg index_reg;

    tcg_target_long disp;



    tcg_prepare_user_ldst(s, &addr_reg, &index_reg, &disp);

    tcg_out_qemu_ld_direct(s, opc, data_reg, addr_reg, index_reg, disp);

#endif

}
