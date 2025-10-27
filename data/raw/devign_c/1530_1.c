int show_bsfs(void *optctx, const char *opt, const char *arg)

{

    AVBitStreamFilter *bsf = NULL;



    printf("Bitstream filters:\n");

    while ((bsf = av_bitstream_filter_next(bsf)))

        printf("%s\n", bsf->name);

    printf("\n");

    return 0;

}
