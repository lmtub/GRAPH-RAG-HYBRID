static int match_format(const char *name, const char *names)

{

    const char *p;

    int len, namelen;



    if (!name || !names)

        return 0;



    namelen = strlen(name);

    while ((p = strchr(names, ','))) {

        len = FFMAX(p - names, namelen);

        if (!av_strncasecmp(name, names, len))

            return 1;

        names = p + 1;

    }

    return !av_strcasecmp(name, names);

}
