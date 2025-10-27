static inline void tcg_reg_sync(TCGContext *s, int reg)

{

    TCGTemp *ts;

    int temp;



    temp = s->reg_to_temp[reg];

    ts = &s->temps[temp];

    assert(ts->val_type == TEMP_VAL_REG);

    if (!ts->mem_coherent && !ts->fixed_reg) {

        if (!ts->mem_allocated) {

            temp_allocate_frame(s, temp);

        }

        tcg_out_st(s, ts->type, reg, ts->mem_reg, ts->mem_offset);

    }

    ts->mem_coherent = 1;

}
