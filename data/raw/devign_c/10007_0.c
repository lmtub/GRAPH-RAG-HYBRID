static void tci_out_label(TCGContext *s, TCGLabel *label)

{

    if (label->has_value) {

        tcg_out_i(s, label->u.value);

        assert(label->u.value);

    } else {

        tcg_out_reloc(s, s->code_ptr, sizeof(tcg_target_ulong), label, 0);

        s->code_ptr += sizeof(tcg_target_ulong);

    }

}
