static int select_frame(AVFilterContext *ctx, AVFilterBufferRef *picref)

{

    SelectContext *select = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];

    double res;



    if (isnan(select->var_values[VAR_START_PTS]))

        select->var_values[VAR_START_PTS] = TS2D(picref->pts);

    if (isnan(select->var_values[VAR_START_T]))

        select->var_values[VAR_START_T] = TS2D(picref->pts) * av_q2d(inlink->time_base);



    select->var_values[VAR_PTS] = TS2D(picref->pts);

    select->var_values[VAR_T  ] = picref->pts * av_q2d(inlink->time_base);

    select->var_values[VAR_POS] = picref->pos == -1 ? NAN : picref->pos;

    select->var_values[VAR_PREV_PTS] = TS2D(picref ->pts);



    select->var_values[VAR_INTERLACE_TYPE] =

        !picref->video->interlaced     ? INTERLACE_TYPE_P :

        picref->video->top_field_first ? INTERLACE_TYPE_T : INTERLACE_TYPE_B;

    select->var_values[VAR_PICT_TYPE] = picref->video->pict_type;



    res = av_expr_eval(select->expr, select->var_values, NULL);

    av_log(inlink->dst, AV_LOG_DEBUG,

           "n:%d pts:%d t:%f pos:%d interlace_type:%c key:%d pict_type:%c "

           "-> select:%f\n",

           (int)select->var_values[VAR_N],

           (int)select->var_values[VAR_PTS],

           select->var_values[VAR_T],

           (int)select->var_values[VAR_POS],

           select->var_values[VAR_INTERLACE_TYPE] == INTERLACE_TYPE_P ? 'P' :

           select->var_values[VAR_INTERLACE_TYPE] == INTERLACE_TYPE_T ? 'T' :

           select->var_values[VAR_INTERLACE_TYPE] == INTERLACE_TYPE_B ? 'B' : '?',

           (int)select->var_values[VAR_KEY],

           av_get_picture_type_char(select->var_values[VAR_PICT_TYPE]),

           res);



    select->var_values[VAR_N] += 1.0;



    if (res) {

        select->var_values[VAR_PREV_SELECTED_N]   = select->var_values[VAR_N];

        select->var_values[VAR_PREV_SELECTED_PTS] = select->var_values[VAR_PTS];

        select->var_values[VAR_PREV_SELECTED_T]   = select->var_values[VAR_T];

        select->var_values[VAR_SELECTED_N] += 1.0;

    }

    return res;

}
