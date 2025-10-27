static int to_integer(char *p, int len)

{

    int ret;

    char *q = av_malloc(sizeof(char) * len);

    if (!q) return -1;

    strncpy(q, p, len);

    ret = atoi(q);

    av_free(q);

    return ret;

}
