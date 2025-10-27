static int ffserver_parse_config_feed(FFServerConfig *config, const char *cmd, const char **p,

                                      FFServerStream **pfeed)

{

    FFServerStream *feed;

    char arg[1024];

    av_assert0(pfeed);

    feed = *pfeed;

    if (!av_strcasecmp(cmd, "<Feed")) {

        char *q;

        FFServerStream *s;

        feed = av_mallocz(sizeof(FFServerStream));

        if (!feed)

            return AVERROR(ENOMEM);

        ffserver_get_arg(feed->filename, sizeof(feed->filename), p);

        q = strrchr(feed->filename, '>');

        if (*q)

            *q = '\0';



        for (s = config->first_feed; s; s = s->next) {

            if (!strcmp(feed->filename, s->filename))

                ERROR("Feed '%s' already registered\n", s->filename);

        }



        feed->fmt = av_guess_format("ffm", NULL, NULL);

        /* default feed file */

        snprintf(feed->feed_filename, sizeof(feed->feed_filename),

                 "/tmp/%s.ffm", feed->filename);

        feed->feed_max_size = 5 * 1024 * 1024;

        feed->is_feed = 1;

        feed->feed = feed; /* self feeding :-) */

        *pfeed = feed;

        return 0;

    }

    av_assert0(feed);

    if (!av_strcasecmp(cmd, "Launch")) {

        int i;



        feed->child_argv = av_mallocz(64 * sizeof(char *));

        if (!feed->child_argv)

            return AVERROR(ENOMEM);

        for (i = 0; i < 62; i++) {

            ffserver_get_arg(arg, sizeof(arg), p);

            if (!arg[0])

                break;



            feed->child_argv[i] = av_strdup(arg);

            if (!feed->child_argv[i])

                return AVERROR(ENOMEM);

        }



        feed->child_argv[i] =

            av_asprintf("http://%s:%d/%s",

                        (config->http_addr.sin_addr.s_addr == INADDR_ANY) ? "127.0.0.1" :

                        inet_ntoa(config->http_addr.sin_addr), ntohs(config->http_addr.sin_port),

                        feed->filename);

        if (!feed->child_argv[i])

            return AVERROR(ENOMEM);

    } else if (!av_strcasecmp(cmd, "ACL")) {

        ffserver_parse_acl_row(NULL, feed, NULL, *p, config->filename,

                config->line_num);

    } else if (!av_strcasecmp(cmd, "File") || !av_strcasecmp(cmd, "ReadOnlyFile")) {

        ffserver_get_arg(feed->feed_filename, sizeof(feed->feed_filename), p);

        feed->readonly = !av_strcasecmp(cmd, "ReadOnlyFile");

    } else if (!av_strcasecmp(cmd, "Truncate")) {

        ffserver_get_arg(arg, sizeof(arg), p);

        /* assume Truncate is true in case no argument is specified */

        if (!arg[0]) {

            feed->truncate = 1;

        } else {

            WARNING("Truncate N syntax in configuration file is deprecated, "

                    "use Truncate alone with no arguments\n");

            feed->truncate = strtod(arg, NULL);

        }

    } else if (!av_strcasecmp(cmd, "FileMaxSize")) {

        char *p1;

        double fsize;



        ffserver_get_arg(arg, sizeof(arg), p);

        p1 = arg;

        fsize = strtod(p1, &p1);

        switch(av_toupper(*p1)) {

        case 'K':

            fsize *= 1024;

            break;

        case 'M':

            fsize *= 1024 * 1024;

            break;

        case 'G':

            fsize *= 1024 * 1024 * 1024;

            break;

        default:

            ERROR("Invalid file size: %s\n", arg);

            break;

        }

        feed->feed_max_size = (int64_t)fsize;

        if (feed->feed_max_size < FFM_PACKET_SIZE*4)

            ERROR("Feed max file size is too small, must be at least %d\n",

                    FFM_PACKET_SIZE*4);

    } else if (!av_strcasecmp(cmd, "</Feed>")) {

        *pfeed = NULL;

    } else {

        ERROR("Invalid entry '%s' inside <Feed></Feed>\n", cmd);

    }

    return 0;

}
