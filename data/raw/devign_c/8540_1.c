static void yop_next_macroblock(YopDecContext *s)

{

    // If we are advancing to the next row of macroblocks

    if (s->row_pos == s->frame.linesize[0] - 2) {

        s->dstptr  += s->frame.linesize[0];

        s->row_pos =  0;

    }else {

        s->row_pos += 2;

    }

    s->dstptr += 2;

}
