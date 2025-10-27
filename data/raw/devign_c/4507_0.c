static void tcg_out_bc(TCGContext *s, int bc, int label_index)

{

    TCGLabel *l = &s->labels[label_index];



    if (l->has_value) {

        tcg_out32(s, bc | reloc_pc14_val(s->code_ptr, l->u.value_ptr));

    } else {

        tcg_out_reloc(s, s->code_ptr, R_PPC_REL14, label_index, 0);

        tcg_out_bc_noaddr(s, bc);

    }

}
