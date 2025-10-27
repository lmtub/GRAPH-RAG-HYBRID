static char *make_digest_auth(HTTPAuthState *state, const char *username,

                              const char *password, const char *uri,

                              const char *method)

{

    DigestParams *digest = &state->digest_params;

    int len;

    uint32_t cnonce_buf[2];

    char cnonce[17];

    char nc[9];

    int i;

    char A1hash[33], A2hash[33], response[33];

    struct AVMD5 *md5ctx;

    uint8_t hash[16];

    char *authstr;



    digest->nc++;

    snprintf(nc, sizeof(nc), "%08x", digest->nc);



    /* Generate a client nonce. */

    for (i = 0; i < 2; i++)

        cnonce_buf[i] = av_get_random_seed();

    ff_data_to_hex(cnonce, (const uint8_t*) cnonce_buf, sizeof(cnonce_buf), 1);

    cnonce[2*sizeof(cnonce_buf)] = 0;



    md5ctx = av_md5_alloc();

    if (!md5ctx)

        return NULL;



    av_md5_init(md5ctx);

    update_md5_strings(md5ctx, username, ":", state->realm, ":", password, NULL);

    av_md5_final(md5ctx, hash);

    ff_data_to_hex(A1hash, hash, 16, 1);

    A1hash[32] = 0;



    if (!strcmp(digest->algorithm, "") || !strcmp(digest->algorithm, "MD5")) {

    } else if (!strcmp(digest->algorithm, "MD5-sess")) {

        av_md5_init(md5ctx);

        update_md5_strings(md5ctx, A1hash, ":", digest->nonce, ":", cnonce, NULL);

        av_md5_final(md5ctx, hash);

        ff_data_to_hex(A1hash, hash, 16, 1);

        A1hash[32] = 0;

    } else {

        /* Unsupported algorithm */

        av_free(md5ctx);

        return NULL;

    }



    av_md5_init(md5ctx);

    update_md5_strings(md5ctx, method, ":", uri, NULL);

    av_md5_final(md5ctx, hash);

    ff_data_to_hex(A2hash, hash, 16, 1);

    A2hash[32] = 0;



    av_md5_init(md5ctx);

    update_md5_strings(md5ctx, A1hash, ":", digest->nonce, NULL);

    if (!strcmp(digest->qop, "auth") || !strcmp(digest->qop, "auth-int")) {

        update_md5_strings(md5ctx, ":", nc, ":", cnonce, ":", digest->qop, NULL);

    }

    update_md5_strings(md5ctx, ":", A2hash, NULL);

    av_md5_final(md5ctx, hash);

    ff_data_to_hex(response, hash, 16, 1);

    response[32] = 0;



    av_free(md5ctx);



    if (!strcmp(digest->qop, "") || !strcmp(digest->qop, "auth")) {

    } else if (!strcmp(digest->qop, "auth-int")) {

        /* qop=auth-int not supported */

        return NULL;

    } else {

        /* Unsupported qop value. */

        return NULL;

    }



    len = strlen(username) + strlen(state->realm) + strlen(digest->nonce) +

              strlen(uri) + strlen(response) + strlen(digest->algorithm) +

              strlen(digest->opaque) + strlen(digest->qop) + strlen(cnonce) +

              strlen(nc) + 150;



    authstr = av_malloc(len);

    if (!authstr)

        return NULL;

    snprintf(authstr, len, "Authorization: Digest ");



    /* TODO: Escape the quoted strings properly. */

    av_strlcatf(authstr, len, "username=\"%s\"",   username);

    av_strlcatf(authstr, len, ",realm=\"%s\"",     state->realm);

    av_strlcatf(authstr, len, ",nonce=\"%s\"",     digest->nonce);

    av_strlcatf(authstr, len, ",uri=\"%s\"",       uri);

    av_strlcatf(authstr, len, ",response=\"%s\"",  response);

    if (digest->algorithm[0])

        av_strlcatf(authstr, len, ",algorithm=%s",  digest->algorithm);

    if (digest->opaque[0])

        av_strlcatf(authstr, len, ",opaque=\"%s\"", digest->opaque);

    if (digest->qop[0]) {

        av_strlcatf(authstr, len, ",qop=\"%s\"",    digest->qop);

        av_strlcatf(authstr, len, ",cnonce=\"%s\"", cnonce);

        av_strlcatf(authstr, len, ",nc=%s",         nc);

    }



    av_strlcatf(authstr, len, "\r\n");



    return authstr;

}
