static void tcg_reg_sync(TCGContext *s, TCGReg reg, TCGRegSet allocated_regs)

{

    TCGTemp *ts = s->reg_to_temp[reg];



    assert(ts->val_type == TEMP_VAL_REG);

    if (!ts->mem_coherent && !ts->fixed_reg) {

        if (!ts->mem_allocated) {

            temp_allocate_frame(s, temp_idx(s, ts));

        } else if (ts->indirect_reg) {

            tcg_regset_set_reg(allocated_regs, ts->reg);

            temp_load(s, ts->mem_base,

                      tcg_target_available_regs[TCG_TYPE_PTR],

                      allocated_regs);

        }

        tcg_out_st(s, ts->type, reg, ts->mem_base->reg, ts->mem_offset);

    }

    ts->mem_coherent = 1;

}
