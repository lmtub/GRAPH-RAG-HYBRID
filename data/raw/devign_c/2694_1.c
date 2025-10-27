static int sami_paragraph_to_ass(AVCodecContext *avctx, const char *src)
{
    SAMIContext *sami = avctx->priv_data;
    int ret = 0;
    char *tag = NULL;
    char *dupsrc = av_strdup(src);
    char *p = dupsrc;
    AVBPrint *dst_content = &sami->encoded_content;
    AVBPrint *dst_source = &sami->encoded_source;
    av_bprint_clear(&sami->encoded_content);
    av_bprint_clear(&sami->content);
    av_bprint_clear(&sami->encoded_source);
    for (;;) {
        char *saveptr = NULL;
        int prev_chr_is_space = 0;
        AVBPrint *dst = &sami->content;
        /* parse & extract paragraph tag */
        p = av_stristr(p, "<P");
        if (!p)
            break;
        if (p[2] != '>' && !av_isspace(p[2])) { // avoid confusion with tags such as <PRE>
            p++;
            continue;
        }
        if (dst->len) // add a separator with the previous paragraph if there was one
            av_bprintf(dst, "\\N");
        tag = av_strtok(p, ">", &saveptr);
        if (!tag || !saveptr)
            break;
        p = saveptr;
        /* check if the current paragraph is the "source" (speaker name) */
        if (av_stristr(tag, "ID=Source") || av_stristr(tag, "ID=\"Source\"")) {
            dst = &sami->source;
            av_bprint_clear(dst);
        }
        /* if empty event -> skip subtitle */
        while (av_isspace(*p))
            p++;
        if (!strncmp(p, "&nbsp;", 6)) {
            ret = -1;
            goto end;
        }
        /* extract the text, stripping most of the tags */
        while (*p) {
            if (*p == '<') {
                if (!av_strncasecmp(p, "<P", 2) && (p[2] == '>' || av_isspace(p[2])))
                    break;
            }
            if (!av_strncasecmp(p, "<BR", 3)) {
                av_bprintf(dst, "\\N");
                p++;
                while (*p && *p != '>')
                    p++;
                if (!*p)
                    break;
                if (*p == '>')
                    p++;
                continue;
            }
            if (!av_isspace(*p))
                av_bprint_chars(dst, *p, 1);
            else if (!prev_chr_is_space)
                av_bprint_chars(dst, ' ', 1);
            prev_chr_is_space = av_isspace(*p);
            p++;
        }
    }
    av_bprint_clear(&sami->full);
    if (sami->source.len) {
        ret = ff_htmlmarkup_to_ass(avctx, dst_source, sami->source.str);
        if (ret < 0)
            goto end;
        av_bprintf(&sami->full, "{\\i1}%s{\\i0}\\N", sami->encoded_source.str);
    }
    ret = ff_htmlmarkup_to_ass(avctx, dst_content, sami->content.str);
    if (ret < 0)
        goto end;
    av_bprintf(&sami->full, "%s", sami->encoded_content.str);
end:
    av_free(dupsrc);
    return ret;
}