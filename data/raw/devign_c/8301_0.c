int ff_get_line(AVIOContext *s, char *buf, int maxlen)

{

    int i = 0;

    char c;



    do {

        c = avio_r8(s);

        if (c && i < maxlen-1)

            buf[i++] = c;

    } while (c != '\n' && c != '\r' && c);

    if (c == '\r' && avio_r8(s) != '\n')

        avio_skip(s, -1);



    buf[i] = 0;

    return i;

}
