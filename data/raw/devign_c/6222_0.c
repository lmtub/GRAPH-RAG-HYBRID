static void mov_text_text_cb(void *priv, const char *text, int len)

{

    MovTextContext *s = priv;

    av_strlcpy(s->ptr, text, FFMIN(s->end - s->ptr, len + 1));

    s->ptr += len;

}
