static inline void tcg_out_goto_label(TCGContext *s, int cond, int label_index)

{

    TCGLabel *l = &s->labels[label_index];



    if (l->has_value) {

        tcg_out_goto(s, cond, l->u.value_ptr);

    } else {

        tcg_out_reloc(s, s->code_ptr, R_ARM_PC24, label_index, 0);

        tcg_out_b_noaddr(s, cond);

    }

}
