int gen_new_label(void)

{

    TCGContext *s = &tcg_ctx;

    int idx;

    TCGLabel *l;



    if (s->nb_labels >= TCG_MAX_LABELS)

        tcg_abort();

    idx = s->nb_labels++;

    l = &s->labels[idx];

    l->has_value = 0;

    l->u.first_reloc = NULL;

    return idx;

}
