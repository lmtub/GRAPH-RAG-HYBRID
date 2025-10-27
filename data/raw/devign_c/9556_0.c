static int64_t update_scr(AVFormatContext *ctx,int stream_index,int64_t pts)

{

    MpegMuxContext *s = ctx->priv_data;

    int64_t scr;



    if (s->is_vcd)

        /* Since the data delivery rate is constant, SCR is computed

           using the formula C + i * 1200 where C is the start constant

           and i is the pack index.

           It is recommended that SCR 0 is at the beginning of the VCD front

           margin (a sequence of empty Form 2 sectors on the CD).

           It is recommended that the front margin is 30 sectors long, so

           we use C = 30*1200 = 36000

           (Note that even if the front margin is not 30 sectors the file

           will still be correct according to the standard. It just won't have

           the "recommended" value).*/

        scr = 36000 + s->packet_number * 1200;

    else {

        /* XXX I believe this calculation of SCR is wrong. SCR

           specifies at which time the data should enter the decoder.

           Two packs cannot enter the decoder at the same time. */



        /* XXX: system clock should be computed precisely, especially for

        CBR case. The current mode gives at least something coherent */

        if (stream_index == s->scr_stream_index

            && pts != AV_NOPTS_VALUE)

            scr = pts;

        else

            scr = s->last_scr;

    }



    s->last_scr=scr;



    return scr;

}    
