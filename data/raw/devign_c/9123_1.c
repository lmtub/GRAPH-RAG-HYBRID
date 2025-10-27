static int glyph_enu_free(void *opaque, void *elem)
{
    av_free(elem);
    return 0;
}