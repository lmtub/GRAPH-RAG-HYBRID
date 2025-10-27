static int handle_connect_error(URLContext *s, const char *desc)

{

    RTMPContext *rt = s->priv_data;

    char buf[300], *ptr, authmod[15];

    int i = 0, ret = 0;

    const char *user = "", *salt = "", *opaque = NULL,

               *challenge = NULL, *cptr = NULL, *nonce = NULL;



    if (!(cptr = strstr(desc, "authmod=adobe")) &&

        !(cptr = strstr(desc, "authmod=llnw"))) {

        av_log(s, AV_LOG_ERROR,

               "Unknown connect error (unsupported authentication method?)\n");

        return AVERROR_UNKNOWN;

    }

    cptr += strlen("authmod=");

    while (*cptr && *cptr != ' ' && i < sizeof(authmod) - 1)

        authmod[i++] = *cptr++;

    authmod[i] = '\0';



    if (!rt->username[0] || !rt->password[0]) {

        av_log(s, AV_LOG_ERROR, "No credentials set\n");

        return AVERROR_UNKNOWN;

    }



    if (strstr(desc, "?reason=authfailed")) {

        av_log(s, AV_LOG_ERROR, "Incorrect username/password\n");

        return AVERROR_UNKNOWN;

    } else if (strstr(desc, "?reason=nosuchuser")) {

        av_log(s, AV_LOG_ERROR, "Incorrect username\n");

        return AVERROR_UNKNOWN;

    }



    if (rt->auth_tried) {

        av_log(s, AV_LOG_ERROR, "Authentication failed\n");

        return AVERROR_UNKNOWN;

    }



    rt->auth_params[0] = '\0';



    if (strstr(desc, "code=403 need auth")) {

        snprintf(rt->auth_params, sizeof(rt->auth_params),

                 "?authmod=%s&user=%s", authmod, rt->username);

        return 0;

    }



    if (!(cptr = strstr(desc, "?reason=needauth"))) {

        av_log(s, AV_LOG_ERROR, "No auth parameters found\n");

        return AVERROR_UNKNOWN;

    }



    av_strlcpy(buf, cptr + 1, sizeof(buf));

    ptr = buf;



    while (ptr) {

        char *next  = strchr(ptr, '&');

        char *value = strchr(ptr, '=');

        if (next)

            *next++ = '\0';

        if (value)

            *value++ = '\0';

        if (!strcmp(ptr, "user")) {

            user = value;

        } else if (!strcmp(ptr, "salt")) {

            salt = value;

        } else if (!strcmp(ptr, "opaque")) {

            opaque = value;

        } else if (!strcmp(ptr, "challenge")) {

            challenge = value;

        } else if (!strcmp(ptr, "nonce")) {

            nonce = value;

        }

        ptr = next;

    }



    if (!strcmp(authmod, "adobe")) {

        if ((ret = do_adobe_auth(rt, user, salt, opaque, challenge)) < 0)

            return ret;

    } else {

        if ((ret = do_llnw_auth(rt, user, nonce)) < 0)

            return ret;

    }



    rt->auth_tried = 1;

    return 0;

}
