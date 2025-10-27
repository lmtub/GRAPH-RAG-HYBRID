static void ini_print_section_header(WriterContext *wctx)

{

    INIContext *ini = wctx->priv;

    AVBPrint buf;

    int i;

    const struct section *section = wctx->section[wctx->level];

    const struct section *parent_section = wctx->level ?

        wctx->section[wctx->level-1] : NULL;



    av_bprint_init(&buf, 1, AV_BPRINT_SIZE_UNLIMITED);

    if (wctx->level == 0) {

        printf("# ffprobe output\n\n");

        return;

    }



    if (wctx->nb_item[wctx->level-1])

        printf("\n");



    for (i = 1; i <= wctx->level; i++) {

        if (ini->hierarchical ||

            !(section->flags & (SECTION_FLAG_IS_ARRAY|SECTION_FLAG_IS_WRAPPER)))

            av_bprintf(&buf, "%s%s", i>1 ? "." : "", wctx->section[i]->name);

    }



    if (parent_section->flags & SECTION_FLAG_IS_ARRAY) {

        int n = parent_section->id == SECTION_ID_PACKETS_AND_FRAMES ?

            wctx->nb_section_packet_frame : wctx->nb_item[wctx->level-1];

        av_bprintf(&buf, ".%d", n);

    }



    if (!(section->flags & (SECTION_FLAG_IS_ARRAY|SECTION_FLAG_IS_WRAPPER)))

        printf("[%s]\n", buf.str);

    av_bprint_finalize(&buf, NULL);

}
