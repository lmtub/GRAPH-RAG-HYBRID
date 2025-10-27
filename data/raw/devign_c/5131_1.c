static int init(AVCodecParserContext *s)

{

    H264Context *h = s->priv_data;

    h->thread_context[0] = h;


    return 0;

}