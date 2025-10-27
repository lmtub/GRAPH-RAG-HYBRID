static void flat_print_section_header(WriterContext *wctx)

{

    FlatContext *flat = wctx->priv;

    AVBPrint *buf = &flat->section_header[wctx->level];

    int i;



    /* build section header */

    av_bprint_clear(buf);

    for (i = 1; i <= wctx->level; i++) {

        if (flat->hierarchical ||

            !(wctx->section[i]->flags & (SECTION_FLAG_IS_ARRAY|SECTION_FLAG_IS_WRAPPER)))

            av_bprintf(buf, "%s%s", wctx->section[i]->name, flat->sep_str);

    }

}
