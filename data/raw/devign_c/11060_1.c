static void swap_sample_fmts_on_filter(AVFilterContext *filter)

{

    AVFilterLink *link = NULL;

    int format, bps;

    int i, j;



    for (i = 0; i < filter->nb_inputs; i++) {

        link = filter->inputs[i];



        if (link->type == AVMEDIA_TYPE_AUDIO &&

            link->out_formats->format_count == 1)

            break;

    }

    if (i == filter->nb_inputs)

        return;



    format = link->out_formats->formats[0];

    bps    = av_get_bytes_per_sample(format);



    for (i = 0; i < filter->nb_outputs; i++) {

        AVFilterLink *outlink = filter->outputs[i];

        int best_idx, best_score = INT_MIN;



        if (outlink->type != AVMEDIA_TYPE_AUDIO ||

            outlink->in_formats->format_count < 2)

            continue;



        for (j = 0; j < outlink->in_formats->format_count; j++) {

            int out_format = outlink->in_formats->formats[j];

            int out_bps    = av_get_bytes_per_sample(out_format);

            int score;



            if (av_get_packed_sample_fmt(out_format) == format ||

                av_get_planar_sample_fmt(out_format) == format) {

                best_idx   = j;

                break;

            }



            /* for s32 and float prefer double to prevent loss of information */

            if (bps == 4 && out_bps == 8) {

                best_idx = j;

                break;

            }



            /* prefer closest higher or equal bps */

            score = -abs(out_bps - bps);

            if (out_bps >= bps)

                score += INT_MAX/2;



            if (score > best_score) {

                best_score = score;

                best_idx   = j;

            }

        }

        FFSWAP(int, outlink->in_formats->formats[0],

               outlink->in_formats->formats[best_idx]);

    }

}
