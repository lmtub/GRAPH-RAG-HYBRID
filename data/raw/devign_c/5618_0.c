static void mov_text_new_line_cb(void *priv, int forced)

{

    MovTextContext *s = priv;

    av_strlcpy(s->ptr, "\n", FFMIN(s->end - s->ptr, 2));

    s->ptr++;

}
