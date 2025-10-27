void av_bitstream_filter_close(AVBitStreamFilterContext *bsfc){



    if(bsfc->filter->close)

        bsfc->filter->close(bsfc);

    av_freep(&bsfc->priv_data);

    av_parser_close(bsfc->parser);

    av_free(bsfc);

}