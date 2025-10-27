static int get_cookies(HTTPContext *s, char **cookies, const char *path,

                       const char *domain)

{

    // cookie strings will look like Set-Cookie header field values.  Multiple

    // Set-Cookie fields will result in multiple values delimited by a newline

    int ret = 0;

    char *next, *cookie, *set_cookies = av_strdup(s->cookies), *cset_cookies = set_cookies;



    if (!set_cookies) return AVERROR(EINVAL);



    *cookies = NULL;

    while ((cookie = av_strtok(set_cookies, "\n", &next))) {

        int domain_offset = 0;

        char *param, *next_param, *cdomain = NULL, *cpath = NULL, *cvalue = NULL;

        set_cookies = NULL;



        while ((param = av_strtok(cookie, "; ", &next_param))) {

            cookie = NULL;

            if        (!av_strncasecmp("path=",   param, 5)) {


                cpath = av_strdup(&param[5]);

            } else if (!av_strncasecmp("domain=", param, 7)) {


                cdomain = av_strdup(&param[7]);

            } else if (!av_strncasecmp("secure",  param, 6) ||

                       !av_strncasecmp("comment", param, 7) ||

                       !av_strncasecmp("max-age", param, 7) ||

                       !av_strncasecmp("version", param, 7)) {

                // ignore Comment, Max-Age, Secure and Version

            } else {

                av_free(cvalue);

                cvalue = av_strdup(param);

            }

        }



        // ensure all of the necessary values are valid

        if (!cdomain || !cpath || !cvalue) {

            av_log(s, AV_LOG_WARNING,

                   "Invalid cookie found, no value, path or domain specified\n");

            goto done_cookie;

        }



        // check if the request path matches the cookie path

        if (av_strncasecmp(path, cpath, strlen(cpath)))

            goto done_cookie;



        // the domain should be at least the size of our cookie domain

        domain_offset = strlen(domain) - strlen(cdomain);

        if (domain_offset < 0)

            goto done_cookie;



        // match the cookie domain

        if (av_strcasecmp(&domain[domain_offset], cdomain))

            goto done_cookie;



        // cookie parameters match, so copy the value

        if (!*cookies) {

            if (!(*cookies = av_strdup(cvalue))) {

                ret = AVERROR(ENOMEM);

                goto done_cookie;

            }

        } else {

            char *tmp = *cookies;

            size_t str_size = strlen(cvalue) + strlen(*cookies) + 3;

            if (!(*cookies = av_malloc(str_size))) {

                ret = AVERROR(ENOMEM);

                goto done_cookie;

            }

            snprintf(*cookies, str_size, "%s; %s", tmp, cvalue);

            av_free(tmp);

        }



        done_cookie:



        av_free(cvalue);

        if (ret < 0) {

            if (*cookies) av_freep(cookies);

            av_free(cset_cookies);

            return ret;

        }

    }



    av_free(cset_cookies);



    return 0;

}