static void tcg_out_reloc(TCGContext *s, tcg_insn_unit *code_ptr, int type,

                          int label_index, intptr_t addend)

{

    TCGLabel *l;

    TCGRelocation *r;



    l = &s->labels[label_index];

    if (l->has_value) {

        /* FIXME: This may break relocations on RISC targets that

           modify instruction fields in place.  The caller may not have 

           written the initial value.  */

        patch_reloc(code_ptr, type, l->u.value, addend);

    } else {

        /* add a new relocation entry */

        r = tcg_malloc(sizeof(TCGRelocation));

        r->type = type;

        r->ptr = code_ptr;

        r->addend = addend;

        r->next = l->u.first_reloc;

        l->u.first_reloc = r;

    }

}
