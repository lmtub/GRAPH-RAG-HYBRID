int vc1_decode_entry_point(AVCodecContext *avctx, VC1Context *v, GetBitContext *gb)

{

    int i;



    av_log(avctx, AV_LOG_DEBUG, "Entry point: %08X\n", show_bits_long(gb, 32));

    v->broken_link = get_bits1(gb);

    v->closed_entry = get_bits1(gb);

    v->panscanflag = get_bits1(gb);

    v->refdist_flag = get_bits1(gb);

    v->s.loop_filter = get_bits1(gb);

    v->fastuvmc = get_bits1(gb);

    v->extended_mv = get_bits1(gb);

    v->dquant = get_bits(gb, 2);

    v->vstransform = get_bits1(gb);

    v->overlap = get_bits1(gb);

    v->quantizer_mode = get_bits(gb, 2);



    if(v->hrd_param_flag){

        for(i = 0; i < v->hrd_num_leaky_buckets; i++) {

            skip_bits(gb, 8); //hrd_full[n]

        }

    }



    if(get_bits1(gb)){

        avctx->coded_width = (get_bits(gb, 12)+1)<<1;

        avctx->coded_height = (get_bits(gb, 12)+1)<<1;

    }

    if(v->extended_mv)

        v->extended_dmv = get_bits1(gb);

    if((v->range_mapy_flag = get_bits1(gb))) {

        av_log(avctx, AV_LOG_ERROR, "Luma scaling is not supported, expect wrong picture\n");

        v->range_mapy = get_bits(gb, 3);

    }

    if((v->range_mapuv_flag = get_bits1(gb))) {

        av_log(avctx, AV_LOG_ERROR, "Chroma scaling is not supported, expect wrong picture\n");

        v->range_mapuv = get_bits(gb, 3);

    }



    av_log(avctx, AV_LOG_DEBUG, "Entry point info:\n"

        "BrokenLink=%i, ClosedEntry=%i, PanscanFlag=%i\n"

        "RefDist=%i, Postproc=%i, FastUVMC=%i, ExtMV=%i\n"

        "DQuant=%i, VSTransform=%i, Overlap=%i, Qmode=%i\n",

        v->broken_link, v->closed_entry, v->panscanflag, v->refdist_flag, v->s.loop_filter,

        v->fastuvmc, v->extended_mv, v->dquant, v->vstransform, v->overlap, v->quantizer_mode);



    return 0;

}
